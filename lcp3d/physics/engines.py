"""
Author: Filipe de Avila Belbute Peres
Based on: M. B. Cline, Rigid body simulation with contact and constraints, 2002
"""

import torch

from ..lcp import LCPFunction


class Engine:
    """Base class for stepping engine."""

    def solve_dynamics(
        self,
        dt: float,
        F_ext: torch.Tensor,
        v: torch.Tensor,
        Je: torch.Tensor,
        Jc: torch.Tensor,
        Jf: torch.Tensor,
        M_world: torch.Tensor,
        E: torch.Tensor,
        Coriolis: torch.Tensor,
        mu: torch.Tensor,
        restitutions: torch.Tensor,
    ):  # world: World, dt):
        raise NotImplementedError

    def post_stabilization(
        self,
        v: torch.Tensor,
        M: torch.Tensor,
        Je: torch.Tensor,
        Jc: torch.Tensor,
        restitutions: torch.Tensor,
    ):
        raise NotImplementedError


class QPEngine(Engine):
    """Engine that uses the Quadratic Programming Engine."""

    def __init__(self, max_iter=10):
        self.lcp_solver = LCPFunction
        self.cached_inverse = None
        self.max_iter = max_iter

    def solve_dynamics(
        self, dt, F_ext, v, Je, Jc, Jf, M_world, E, Coriolis, mu, restitutions
    ):  # world: World, dt):
        Nu = M_world.size(0)
        neq = Je.size(0) if Je.ndimension() > 0 else 0
        u = torch.matmul(M_world, v) + dt * F_ext.view(-1) - dt * Coriolis
        if neq > 0:
            u = torch.cat([u, u.new_zeros(neq)], dim=-1)
        if Jc.shape[0] == 0:
            # No contact constraints, no complementarity conditions
            if neq > 0:
                P = torch.cat(
                    [
                        torch.cat([M_world, -Je.T], dim=1),
                        torch.cat([Je, Je.new_zeros(neq, neq)], dim=1),
                    ]
                )
            else:
                P = M_world
            x = torch.linalg.solve(P, u)
        else:
            # Solve Mixed LCP (Kline 2.7.2)
            v = torch.matmul(Jc, v) * restitutions
            h = torch.cat([v, v.new_zeros(Jf.size(0) + mu.size(0))])
            G = torch.cat([Jc, Jf, Jf.new_zeros(mu.size(0), Jf.size(1))])
            F = G.new_zeros(G.size(0), G.size(0))
            F[Jc.size(0) : -E.size(1), -E.size(1) :] = E
            F[-mu.size(0) :, : mu.size(1)] = mu
            F[-mu.size(0) :, mu.size(1) : mu.size(1) + E.size(0)] = -E.T

            b = Je.new_zeros(Je.size(0)).unsqueeze(0)
            Je = Je.unsqueeze(0)
            M = M_world.unsqueeze(0)
            u = u[:Nu].unsqueeze(0)
            G = G.unsqueeze(0)
            h = h.unsqueeze(0)
            F = F.unsqueeze(0)
            x = -self.lcp_solver(max_iter=self.max_iter, verbose=-1)(
                M, u, G, h, Je, b, F
            )
        new_v = x[..., :Nu].squeeze(0)
        return new_v

    def post_stabilization(self, v, M, Je, Jc, restitutions):
        ge = torch.matmul(Je, v)
        u = torch.cat([Je.new_zeros(Je.size(1)), ge])

        if Jc.shape[0] == 0:
            neq = Je.size(0) if Je.ndimension() > 0 else 0
            if neq > 0:
                P = torch.cat(
                    [
                        torch.cat([M, -Je.t()], dim=1),
                        torch.cat([Je, Je.new_zeros(neq, neq)], dim=1),
                    ]
                )
            else:
                P = M
            x = torch.linalg.solve(P, u)
        else:
            gc = torch.matmul(Jc, v) + torch.matmul(Jc, v) * -restitutions
            Je = Je.unsqueeze(0)
            Jc = Jc.unsqueeze(0)
            h = u[: M.size(0)].unsqueeze(0)
            b = u[M.size(0) :].unsqueeze(0)
            M = M.unsqueeze(0)
            gc = gc.unsqueeze(0)
            F = Jc.new_zeros(Jc.size(1), Jc.size(1)).unsqueeze(0)
            x = self.lcp_solver()(M, h, Jc, gc, Je, b, F)
        dp = -x[: M.size(0)]
        return dp
