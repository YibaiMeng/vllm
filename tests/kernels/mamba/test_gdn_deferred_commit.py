# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred-commit GDN MTP kernel (gdn_mtp_cuda_deferred + gdn_mtp_materialize)
vs the snapshot kernel (gdn_mtp_cuda), bitwise, over multi-step sequences.

Pool A runs the snapshot kernel every step; pool B runs the deferred kernel
(D steps), with materialize + snapshot-contract steps mixed in. Each step:
outputs equal bitwise; every state the snapshot contract would read next (the
slot of each request's last accepted token) equals pool A's bitwise after
materializing a copy of pool B; pool B / header bytes change only where the
contract says.
"""

import random
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.model_executor.layers.mamba.ops import gdn_mtp_cuda as gmc  # noqa: E402

K = V = 128
EPS = 1e-6
HDR = gmc.GDN_DEFERRED_HDR_INTS
REC_BYTES = (2 * 8 * 128 + 8) * 4  # rk[8][128], rdl[8][128], rd[8] (fp32)


@pytest.fixture(scope="module", autouse=True)
def _built():
    if not gmc.enable():
        pytest.skip("GDN MTP CUDA kernel did not build")


def _bytes(x: torch.Tensor) -> torch.Tensor:
    """[S, HV, V, K] -> [S, HV, head bytes] uint8 view."""
    return x.view(torch.uint8).reshape(x.shape[0], x.shape[1], -1)


def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
    return torch.equal(
        x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
    )


def _changed_outside(x: torch.Tensor, y: torch.Tensor, allowed: torch.Tensor) -> bool:
    """Any byte of x != y outside the allowed [S, HV, head bytes] mask."""
    return bool(((_bytes(x) != _bytes(y)) & ~allowed).any())


def _boundary(P: int, W: int, bs: int) -> bool:
    """A mamba block boundary m * bs (m >= 1) in [P - W + 1, P + 3W]."""
    if bs <= 0:
        return False
    lo = max(P - W + 1, 1)
    return -(-lo // bs) * bs <= P + 3 * W


class _Sim:
    def __init__(self, state_dtype, H, HV, W, block_size, seed, n_req=None):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.dev = torch.device("cuda")
        self.H, self.HV, self.W, self.bs = H, HV, W, block_size
        self.n_req = n_req or W + 2
        self.max_reqs = self.n_req + 3
        S = 1 + self.n_req * 2 * W + 2
        self.priv = [
            list(range(1 + i * 2 * W, 1 + (i + 1) * 2 * W)) for i in range(self.n_req)
        ]
        mk = lambda: (0.05 * torch.randn(S, HV, V, K, device=self.dev)).to(state_dtype)  # noqa: E731
        self.poolA, self.poolB = mk(), mk()  # different garbage in unwritten slots
        self.hdr = torch.randint(
            -3, 99, (self.max_reqs, HV, HDR), dtype=torch.int32, device=self.dev
        )
        self.hdr[..., 0] = 0
        self.exp_hdr = self.hdr.clone()
        self.rho = self.rng.sample(range(self.max_reqs), self.n_req)
        self.row = [self.rng.sample(p, W) for p in self.priv]
        self.acc = [1] * self.n_req
        self.P = [self.rng.randrange(0, 300) for _ in range(self.n_req)]
        # slots row of a valid header
        self.hrow: list[list[int] | None] = [None] * self.n_req
        for r in self.row:
            self.poolB[r[0]] = self.poolA[r[0]]
        self.A_log = torch.log(torch.empty(HV, device=self.dev).uniform_(1, 16))
        self.dt_bias = torch.randn(HV, device=self.dev)
        self.w = (1 + 0.1 * torch.randn(V, device=self.dev)).to(torch.bfloat16)
        self.scale = K**-0.5
        self.stats: dict[str, Any] = {
            "replay": [0] * (W + 1),
            "fallback": 0,
            "first": 0,
            "neg": 0,
            "plain": 0,
        }

    def _next_row(self, i):
        keep = self.row[i][self.acc[i] - 1]
        others = self.rng.sample([s for s in self.priv[i] if s != keep], self.W - 1)
        others.insert(self.acc[i] - 1, keep)
        return others

    def _materialize_real(self, reqs):
        if not reqs:
            return
        nacc = torch.ones(self.max_reqs, dtype=torch.int32)
        for i in reqs:
            nacc[self.rho[i]] = self.acc[i]
        rows = torch.tensor([self.rho[i] for i in reqs], dtype=torch.int32)
        gmc.gdn_mtp_materialize(
            self.poolB, self.hdr, rows.to(self.dev), nacc.to(self.dev)
        )
        for i in reqs:
            if self.hrow[i] is not None:
                self.exp_hdr[self.rho[i], :, 0] = 0
                self.hrow[i] = None
            s = self.row[i][self.acc[i] - 1]
            assert _same(self.poolB[s], self.poolA[s])
        assert torch.equal(self.hdr, self.exp_hdr)

    def _check_materialize(self):
        """Materialize a copy of pool B for every request row (plus -1 and
        unused rows): each valid header's hdr.slots[n-1] equals pool A's slot.
        """
        cB, ch = self.poolB.clone(), self.hdr.clone()
        nacc = torch.randint(1, self.W + 1, (self.max_reqs,), dtype=torch.int32)
        for i in range(self.n_req):
            nacc[self.rho[i]] = self.acc[i]
        rows = list(range(self.max_reqs)) + [-1]
        self.rng.shuffle(rows)
        gmc.gdn_mtp_materialize(
            cB,
            ch,
            torch.tensor(rows, dtype=torch.int32, device=self.dev),
            nacc.to(self.dev),
        )
        allowed = torch.zeros(_bytes(cB).shape, dtype=torch.bool, device=self.dev)
        exp = self.exp_hdr.clone()
        for i in range(self.n_req):
            hrow = self.hrow[i]
            if hrow is None:
                continue
            s = hrow[self.acc[i] - 1]
            assert _same(cB[s], self.poolA[s]), f"materialized slot of request {i}"
            allowed[s] = True
            exp[self.rho[i], :, 0] = 0
        assert not _changed_outside(cB, self.poolB, allowed)
        assert torch.equal(ch, exp)

    def step(self, s):
        W, HV, dev = self.W, self.HV, self.dev
        plain = s % 9 == 5  # materialize everything, then the snapshot kernel
        sched = list(range(self.n_req))
        if self.rng.random() < 0.4:
            sched.remove(self.rng.choice(sched))  # not scheduled this step
        neg = set() if plain or s % 4 != 2 else {self.rng.choice(sched)}
        self._materialize_real(
            [i for i in range(self.n_req) if self.hrow[i] is not None]
            if plain
            else sorted(neg)
        )
        self.rng.shuffle(sched)
        nxt, lens, rows = {}, [], []
        for i in sched:
            nxt[i] = (i + s) % W + 1  # every n = 1..W in every step
            lens.append(self.rng.randint(nxt[i], W))
            rows.append(self._next_row(i))
        # Padded row (slots 0) pointing at an unscheduled request row (or -1).
        idle = [self.rho[i] for i in range(self.n_req) if i not in sched]
        lens.append(W)
        rows.append([0] * W)
        N = len(rows)
        req_rows = [-1 if i in neg else self.rho[i] for i in sched]
        req_rows += [idle[0] if idle else -1, -5, -5]  # >= N entries
        acc = [self.acc[i] for i in sched] + [1]
        cu = [0]
        for t in lens:
            cu.append(cu[-1] + t)
        L = cu[-1]
        ncomp = torch.randint(0, 9999, (self.max_reqs,), dtype=torch.int32)
        for i in sched:
            ncomp[self.rho[i]] = self.P[i]

        H = self.H
        qkv_w = 2 * H * K + HV * V
        qkv = torch.randn(L, qkv_w + 64, device=dev).to(torch.bfloat16)[:, :qkv_w]
        ba = torch.randn(L, 2 * HV, device=dev).to(torch.bfloat16)
        b, a = ba[:, :HV], ba[:, HV:]
        gate = torch.randn(L, HV, V, device=dev).to(torch.bfloat16)
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)  # noqa: E731
        si, cu_t, acc_t = i32(rows), i32(cu), i32(acc)

        def args(pool, out):
            return (
                qkv,
                a,
                b,
                self.A_log,
                self.dt_bias,
                si,
                cu_t,
                acc_t,
                pool,
                gate,
                self.w,
                out,
                self.scale,
                EPS,
                "silu",
            )

        A0, B0 = self.poolA.clone(), self.poolB.clone()
        outA = torch.full((L, HV, V), 7.0, dtype=torch.bfloat16, device=dev)
        outB = torch.full((L, HV, V), 9.0, dtype=torch.bfloat16, device=dev)
        assert gmc.gdn_mtp_cuda(*args(self.poolA, outA))
        if plain:
            assert gmc.gdn_mtp_cuda(*args(self.poolB, outB))
        else:
            extra = (i32(req_rows), self.hdr, ncomp.to(dev), self.bs)
            assert gmc.gdn_mtp_cuda_deferred(*(args(self.poolB, outB) + extra))
        assert _same(outA, outB), f"step {s} outputs"
        assert torch.all(outB[cu[N - 1] :] == 0)  # padded row

        allowed = torch.zeros(_bytes(B0).shape, dtype=torch.bool, device=dev)
        for k, i in enumerate(sched):
            row, T, n = rows[k], lens[k], acc[k]
            snap = plain or i in neg or _boundary(self.P[i], W, self.bs)
            if plain:
                self.stats["plain"] += 1
            elif i in neg:
                self.stats["neg"] += 1
            elif snap:
                self.stats["fallback"] += 1
            if snap:
                for t in range(T):
                    allowed[row[t]] = True
                    assert _same(self.poolB[row[t]], self.poolA[row[t]]), (
                        f"step {s} snapshot {t}"
                    )
                if not plain and i not in neg and self.hrow[i] is not None:
                    self.exp_hdr[self.rho[i], :, 0] = 0
                self.hrow[i] = None
            else:
                if self.hrow[i] is not None:
                    self.stats["replay"][n] += 1
                else:
                    self.stats["first"] += 1
                if self.hrow[i] is not None or n != 1:
                    allowed[row[0]] = True
                # Committed state S(P) == pool A's snapshot source.
                assert _same(self.poolB[row[0]], A0[row[n - 1]]), f"step {s} commit"
                allowed[row[W - 1], :, :REC_BYTES] = True
                self.exp_hdr[self.rho[i], :, :12] = i32([1, W] + row + [0] * (10 - W))
                self.hrow[i] = row
        assert not _changed_outside(self.poolB, B0, allowed), f"step {s} bytes"
        assert torch.equal(self.hdr, self.exp_hdr), f"step {s} header"

        for k, i in enumerate(sched):
            self.row[i] = rows[k]
            self.acc[i] = nxt[i]
            self.P[i] += nxt[i]
        self._check_materialize()


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W", [5, 4])
@pytest.mark.parametrize("H,HV", [(2, 4), (2, 2)])
@pytest.mark.parametrize("block_size", [0, 64])
def test_deferred_matches_snapshot_kernel(state_dtype, W, H, HV, block_size):
    sim = _Sim(state_dtype, H, HV, W, block_size, seed=W * 100 + HV * 10 + block_size)
    for s in range(36):
        sim.step(s)
    st = sim.stats
    assert all(st["replay"][n] > 0 for n in range(1, W + 1)), st
    assert st["first"] > 0 and st["neg"] > 0 and st["plain"] > 0, st
    assert (st["fallback"] > 0) == (block_size > 0), st
    # Last: D -> snapshot via materialize, then the snapshot kernel continues.
    sim._materialize_real(list(range(sim.n_req)))
    assert not torch.any(sim.hdr[..., 0])
    sim.step(5)


def test_deferred_invalid_source_and_layout():
    dev = torch.device("cuda")
    H, HV, L = 1, 2, 4
    qkv = torch.randn(L, 2 * H * K + HV * V, device=dev).to(torch.bfloat16)
    ab = torch.randn(L, HV, device=dev).to(torch.bfloat16)
    state = torch.randn(6, HV, V, K, device=dev)
    gate = torch.randn(L, HV, V, device=dev).to(torch.bfloat16)
    vec = torch.zeros(HV, device=dev)
    w = torch.ones(V, device=dev)
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)  # noqa: E731
    # Invalid source: header invalid and slots[num_accepted - 1] == 0.
    si, cu, acc = i32([[1, 0, 2, 3]]), i32([0, L]), i32([2])
    hdr = torch.zeros(3, HV, HDR, dtype=torch.int32, device=dev)
    hdr[..., 1:] = 4
    rows, ncomp = i32([1]), i32([0, 0, 0])
    st0, hdr0 = state.clone(), hdr.clone()
    out = torch.full((L, HV, V), 7.0, device=dev, dtype=torch.bfloat16)

    def run(*extra, st=state):
        return gmc.gdn_mtp_cuda_deferred(
            qkv,
            ab,
            ab,
            vec,
            vec,
            si,
            cu,
            acc,
            st,
            gate,
            w,
            out,
            1.0,
            EPS,
            "silu",
            *extra,
        )

    assert run(rows, hdr, ncomp, 0)
    assert torch.all(out == 0)
    assert torch.equal(state, st0) and torch.equal(hdr, hdr0)
    # Layout contract violations launch nothing.
    out.fill_(7.0)
    bad = [
        (rows, hdr[..., :8].contiguous(), ncomp, 0),
        (rows, hdr.long(), ncomp, 0),
        (rows.long(), hdr, ncomp, 0),
        (i32([]), hdr, ncomp, 0),
        (rows, hdr, i32([0, 0]), 0),
        (rows, hdr, ncomp, -1),
    ]
    for extra in bad:
        assert not run(*extra)
    assert not run(rows, hdr, ncomp, 0, st=torch.zeros(6, HV, V, 64, device=dev))
    assert torch.all(out == 7.0)
    with pytest.raises(ValueError):
        gmc.gdn_mtp_materialize(state, hdr[:, :1], rows, i32([1, 1, 1]))
