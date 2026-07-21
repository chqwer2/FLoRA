"""
Self-contained LeNA core for the probe experiments (no peft/transformers needed).

The math here MIRRORS Experiments/peft/tuners/lena/{activations,gates,layer}.py:
  - learnable per-dim spline activation  (FlexSpline, identity-init)
  - static selection gate g in [0,1]      (Gate: sigmoid soft / hard-STE)
  - gated interpolation in the code:  h = z + g*(phi(norm(z)) - z)
  - output:  y = W x + s * B h            (W frozen, B zero-init)

Kept standalone so the decisive rank/sparsity experiments run on CPU in seconds,
independent of the (currently version-clashed) peft environment.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Learnable per-dim spline  (mirror of FlexSpline, mode="dim", identity init)
# ----------------------------------------------------------------------
class LearnableSpline1D(nn.Module):
    def __init__(self, dim: int, K: int = 16, x_min: float = -3.0, x_max: float = 3.0,
                 init_eps: float = 1e-3):
        super().__init__()
        self.K, self.x_min, self.x_max = K, x_min, x_max
        self.register_buffer("kx", torch.linspace(x_min, x_max, K))
        ky = self.kx.view(1, K).repeat(dim, 1).clone()          # identity: y=x at the knots
        ky = ky + torch.randn_like(ky) * init_eps
        self.ky = nn.Parameter(ky)                               # (dim, K)

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # z: (N, dim)
        N, D = z.shape
        zc = z.clamp(self.x_min, self.x_max)
        idx = (torch.bucketize(zc, self.kx) - 1).clamp(0, self.K - 2)   # (N, D)
        x0 = self.kx[idx]
        x1 = self.kx[idx + 1]
        d = torch.arange(D, device=z.device).view(1, D).expand(N, D)
        y0 = self.ky[d, idx]
        y1 = self.ky[d, idx + 1]
        t = (zc - x0) / (x1 - x0 + 1e-12)
        return y0 + t * (y1 - y0)


# ----------------------------------------------------------------------
# Static selection gate  (mirror of Gate: sigmoid soft / hard-STE, per-dim)
# ----------------------------------------------------------------------
class SelectionGate(nn.Module):
    def __init__(self, dim: int, init: float = -2.0, hard: bool = False):
        super().__init__()
        self.theta = nn.Parameter(torch.full((dim,), float(init)))
        self.hard = hard

    def value(self) -> torch.Tensor:
        s = torch.sigmoid(self.theta)
        if self.hard:
            h = (s >= 0.5).to(s.dtype)
            return h.detach() - s.detach() + s     # straight-through
        return s

    def openness(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)


# ----------------------------------------------------------------------
# Adapters over a frozen linear layer W  (d_out x d_in)
# ----------------------------------------------------------------------
class LoRAAdapter(nn.Module):
    def __init__(self, W: torch.Tensor, r: int, alpha: float | None = None):
        super().__init__()
        d_out, d_in = W.shape
        self.register_buffer("W", W)
        self.A = nn.Parameter(torch.empty(r, d_in)); nn.init.xavier_uniform_(self.A)
        self.B = nn.Parameter(torch.zeros(d_out, r))            # zero init => starts at W x
        self.s = (alpha if alpha is not None else r) / r
        self.r = r

    def forward(self, x):
        z = x @ self.A.t()
        return x @ self.W.t() + self.s * (z @ self.B.t())

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LeNAAdapter(nn.Module):
    """LeNA with a learnable spline nonlinearity in the r-dim code.

    gate_mode:
      "on"    -> g = 1 (always nonlinear; isolates the EXPRESSIVITY question, no gate params)
      "soft"  -> learnable soft sigmoid gate
      "hard"  -> learnable hard STE gate  (exact LoRA fallback where closed)
    """
    def __init__(self, W: torch.Tensor, r: int, alpha: float | None = None,
                 K: int = 16, gate_mode: str = "on", gate_init: float = -2.0,
                 norm_before_act: bool = True):
        super().__init__()
        d_out, d_in = W.shape
        self.register_buffer("W", W)
        self.A = nn.Parameter(torch.empty(r, d_in)); nn.init.xavier_uniform_(self.A)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.s = (alpha if alpha is not None else r) / r
        self.r = r
        self.spline = LearnableSpline1D(r, K=K)
        self.norm = nn.LayerNorm(r) if norm_before_act else None
        self.gate_mode = gate_mode
        self.gate = None if gate_mode == "on" else SelectionGate(r, gate_init, hard=(gate_mode == "hard"))

    def forward(self, x):
        z = x @ self.A.t()                                     # (N, r)
        zc = self.norm(z) if self.norm is not None else z
        phi = self.spline(zc)
        if self.gate is None:
            h = phi                                            # g = 1
        else:
            g = self.gate.value().view(1, -1)                  # (1, r)
            h = z + g * (phi - z)                              # skip uses raw z => LoRA fallback
        return x @ self.W.t() + self.s * (h @ self.B.t())

    def gate_openness(self):
        return None if self.gate is None else self.gate.openness()

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BilinearAdapter(nn.Module):
    """Second-order (multiplicative) low-rank update:  Delta(x) = B * ((A1 x) ⊙ (A2 x)).

    Captures pairwise x_i x_j interactions a linear rank-r update cannot. The local
    Jacobian J(x)=B(diag(A1 x)A2 + diag(A2 x)A1) is input-dependent => input-conditional.
    """
    def __init__(self, W: torch.Tensor, r: int, alpha: float | None = None):
        super().__init__()
        d_out, d_in = W.shape
        self.register_buffer("W", W)
        self.A1 = nn.Parameter(torch.empty(r, d_in)); nn.init.xavier_uniform_(self.A1)
        self.A2 = nn.Parameter(torch.empty(r, d_in)); nn.init.xavier_uniform_(self.A2)
        self.B = nn.Parameter(torch.zeros(d_out, r))   # zero init => starts at W x
        self.s = (alpha if alpha is not None else r) / r
        self.r = r

    def forward(self, x):
        z = (x @ self.A1.t()) * (x @ self.A2.t())
        return x @ self.W.t() + self.s * (z @ self.B.t())

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def update_family_rank(adapter, W, d_in, n=64, device="cpu", thresh=0.99):
    """How input-conditional is the learned update? Stack per-input Jacobians of
    Delta(x)=adapter(x)-Wx and return the effective rank of that stack.
    Linear LoRA -> 1 (same update for every input). Nonlinear -> >1.
    """
    import torch.autograd.functional as AF
    X = torch.randn(n, d_in, device=device)

    def delta(xrow):
        return adapter(xrow.unsqueeze(0)).squeeze(0) - xrow @ W.t()

    Js = []
    for i in range(n):
        J = AF.jacobian(delta, X[i], vectorize=True)   # (d_out, d_in)
        Js.append(J.reshape(-1))
    M = torch.stack(Js, 0)                              # (n, d_out*d_in)
    sv = torch.linalg.svdvals(M)
    energy = (sv ** 2)
    cum = torch.cumsum(energy, 0) / energy.sum()
    eff_rank = int((cum < thresh).sum().item()) + 1     # #components for 99% energy
    return eff_rank, sv[0].item(), (sv[min(4, len(sv)-1)].item())


# ----------------------------------------------------------------------
# Synthetic target:  y = W x + s* * B* sigma(A* x)   (a NONLINEAR-low-rank update)
# ----------------------------------------------------------------------
def make_target(d_in=64, d_out=64, r0=4, sigma="relu", s_star=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    W  = torch.randn(d_out, d_in, generator=g) / math.sqrt(d_in)
    As = torch.randn(r0, d_in, generator=g) / math.sqrt(d_in)
    Bs = torch.randn(d_out, r0, generator=g) / math.sqrt(r0)
    acts = {"relu": torch.relu, "tanh": torch.tanh, "sin": torch.sin, "gelu": torch.nn.functional.gelu}
    act = acts[sigma]

    def f(x):
        z = x @ As.t()
        return x @ W.t() + s_star * (act(z) @ Bs.t())

    return W, f


def fit(adapter, f, W, n_train=4096, n_test=2048, d_in=64, steps=1500, lr=3e-3,
        batch=512, seed=1, device="cpu"):
    """Fit adapter to the target f on random Gaussian inputs; return final test MSE."""
    torch.manual_seed(seed)
    adapter = adapter.to(device)
    Xtr = torch.randn(n_train, d_in, device=device)
    Ytr = f(Xtr).detach()
    Xte = torch.randn(n_test, d_in, device=device)
    Yte = f(Xte).detach()
    opt = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=lr)
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch,), device=device)
        pred = adapter(Xtr[idx])
        loss = (pred - Ytr[idx]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        test_mse = (adapter(Xte) - Yte).pow(2).mean().item()
    # normalize by target-update energy so numbers are comparable across seeds
    with torch.no_grad():
        base_mse = (Yte - Xte @ W.t()).pow(2).mean().item()   # error of doing NOTHING (=frozen W)
    return test_mse, base_mse, adapter.n_trainable()
