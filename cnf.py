import torch
from torch import nn
from torchdiffeq import odeint_adjoint

class ODECNF(nn.Module):
    def __init__(self, func, interval=1., tol=1e-6, method="dopri5", nonself_connections=False, energy_regularization=0.0, jacnorm_regularization=0.0) -> None:
        super(ODECNF, self).__init__()
        self.func = func
        self.tol = tol
        self.method = method
        self.nonself_connections = nonself_connections
        self.energy_regularization = energy_regularization
        self.jacnorm_regularization = jacnorm_regularization
        self.interval = interval
        self.ode = _CNFODE(func)

    def forward(self, x, logpx, cond):
        e = torch.randn_like(x)
        energy = torch.zeros(1).to(x)
        jacnorm = torch.zeros(1).to(x)
        initial_state = (e, x, logpx, energy, jacnorm, cond)
        tt = torch.tensor([0., self.interval]).to(x)
        solution = odeint_adjoint(
            self,
            initial_state,
            tt,
            rtol=self.tol,
            atol=self.tol,
            method=self.method,
        )
        _, y, logpy, energy, jacnorm = tuple(s[-1] for s in solution)
        regularization = (
            self.energy_regularization * (energy - energy.detach()) +
            self.jacnorm_regularization * (jacnorm - jacnorm.detach())
        )
        return y, logpy + regularization


class _CNFODE(nn.Module):
    def __init__(self, func) -> None:
        super(_CNFODE, self).__init__()
        self.func = func

    def forward(self, t, state):
        e, x, _, _, _, cond = state
        with torch.enable_grad():
            x = x.requires_grad(True)
            dx = self.func(t, x, cond)
        div, vjp = divergnece(dx, x, e=e if self.training else None)
        d_energy = torch.sum(dx * dx) / x.shape[0]
        d_jacnorm = torch.sum(vjp*vjp) / x.shape[0]
        return torch.zeros_like(e), dx, -div, d_energy, d_jacnorm

def divergnece(f, x, e=None):
    assert f.shape == x.shape
    size = f.shape[1]
    if e is None: # brute-force computation (used only in testing)
        sum_diag = 0
        for i in range(size):
            sum_diag += torch.autograd.grad(f[:, i].sum(), x)[0][:, i]
        div = sum_diag
        div.detach_()
        vjp = torch.zeros(1, device=x.device)
    else:
        e.shape == f.shape
        vjp = torch.autograd.grad(f, x, e, create_graph=True, retain_graph=True)
        div = torch.sum(vjp * e, dim=1)
    return div, vjp