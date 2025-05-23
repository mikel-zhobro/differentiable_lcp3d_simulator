import torch
from torch.autograd import Function

from .util import bger, expandParam, extract_nBatch
from ..config import get_config
from .solvers import batch as pdipm_b
import cvxpy

from enum import Enum

config = get_config()


class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2


def LCPFunction(
    eps=1e-12,
    verbose=0,
    notImprovedLim=3,
    max_iter=20,
    solver=config.solver_id,
    check_Q_spd=True,
):
    class LCPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_, F_):
            """Solve a batch of QPs.

            This function solves a batch of QPs, each optimizing over
            `nz` variables and having `nineq` inequality constraints
            and `neq` equality constraints.
            The optimization problem for each instance in the batch
            (dropping indexing from the notation) is of the form

                \hat z =   argmin_z 1/2 z^T Q z + p^T z
                        subject to Gz <= h
                                    Az  = b

            where Q \in S^{nz,nz},
                S^{nz,nz} is the set of all positive semi-definite matrices,
                p \in R^{nz}
                G \in R^{nineq,nz}
                h \in R^{nineq}
                A \in R^{neq,nz}
                b \in R^{neq}
                F \in R^{nz}

            These parameters should all be passed to this function as
            Variable- or Parameter-wrapped Tensors.
            (See torch.autograd.Variable and torch.nn.parameter.Parameter)

            If you want to solve a batch of QPs where `nz`, `nineq` and `neq`
            are the same, but some of the contents differ across the
            minibatch, you can pass in tensors in the standard way
            where the first dimension indicates the batch example.
            This can be done with some or all of the coefficients.

            You do not need to add an extra dimension to coefficients
            that will not change across all of the minibatch examples.
            This function is able to infer such cases.

            If you don't want to use any equality or inequality constraints,
            you can set the appropriate values to:

                e = Variable(torch.Tensor())

            Parameters:
            Q:  A (nBatch, nz, nz) or (nz, nz) Tensor.
            p:  A (nBatch, nz) or (nz) Tensor.
            G:  A (nBatch, nineq, nz) or (nineq, nz) Tensor.
            h:  A (nBatch, nineq) or (nineq) Tensor.
            A:  A (nBatch, neq, nz) or (neq, nz) Tensor.
            b:  A (nBatch, neq) or (neq) Tensor.
            F:  A (nBatch, nz) or (nz) Tensor.

            Returns: \hat z: a (nBatch, nz) Tensor.
            """
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_, F_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)
            F, _ = expandParam(F_, nBatch, 3)

            if check_Q_spd:
                for i in range(nBatch):
                    e = torch.linalg.eigvals(Q[i])
                    if not torch.all(e.real > 0):
                        raise RuntimeError("Q is not SPD.")

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert neq > 0 or nineq > 0
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            if solver == QPSolvers.PDIPM_BATCHED.value:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A, F)
                zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                    Q,
                    p,
                    G,
                    h,
                    A,
                    b,
                    F,
                    ctx.Q_LU,
                    ctx.S_LU,
                    ctx.R,
                    eps,
                    verbose,
                    notImprovedLim,
                    max_iter,
                )
            elif solver == QPSolvers.CVXPY.value:
                vals = torch.Tensor(nBatch).type_as(Q)
                zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
                lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                nus = (
                    torch.Tensor(nBatch, ctx.neq).type_as(Q)
                    if ctx.neq > 0
                    else torch.Tensor()
                )
                slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                for i in range(nBatch):
                    Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                    vals[i], zhati, nui, lami, si = cvxpy.forward_single_np(
                        *[
                            x.cpu().detach().numpy() if x is not None else None
                            for x in (Q[i], p[i], G[i], h[i], Ai, bi)
                        ]
                    )
                    # if zhati[0] is None:
                    #     import IPython, sys; IPython.embed(); sys.exit(-1)
                    zhats[i] = torch.Tensor(zhati)
                    lams[i] = torch.Tensor(lami)
                    slacks[i] = torch.Tensor(si)
                    if neq > 0:
                        nus[i] = torch.Tensor(nui)

                ctx.vals = vals
                ctx.lams = lams
                ctx.nus = nus
                ctx.slacks = slacks
            else:
                assert False

            ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_, F_)
            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            zhats, Q, p, G, h, A, b, F = ctx.saved_tensors
            nBatch = extract_nBatch(Q, p, G, h, A, b, F)
            Q, Q_e = expandParam(Q, nBatch, 3)
            p, p_e = expandParam(p, nBatch, 2)
            G, G_e = expandParam(G, nBatch, 3)
            h, h_e = expandParam(h, nBatch, 2)
            A, A_e = expandParam(A, nBatch, 3)
            b, b_e = expandParam(b, nBatch, 2)
            F, F_e = expandParam(F, nBatch, 3)

            # neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz
            neq, nineq = ctx.neq, ctx.nineq

            if solver == QPSolvers.CVXPY.value:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A, F)

            # Clamp here to avoid issues coming up when the slacks are too small.
            # TODO: A better fix would be to get lams and slacks from the
            # solver that don't have this issue.
            d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)

            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
            dx, _, dlam, dnu = pdipm_b.solve_kkt(
                ctx.Q_LU,
                d,
                G,
                A,
                ctx.S_LU,
                dl_dzhat,
                torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, nineq).type_as(G),
                torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor(),
            )

            dps = dx
            # Change
            dFs = bger(dlam, ctx.lams)
            # if F_e:
            #     dFs = dFs.mean(0)
            dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
            # if G_e:
            #     dGs = dGs.mean(0)
            dhs = -dlam
            # if h_e:
            #     dhs = dhs.mean(0)
            if neq > 0:
                dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
                dbs = -dnu
                # if A_e:
                #     dAs = dAs.mean(0)
                # if b_e:
                #     dbs = dbs.mean(0)
            else:
                dAs, dbs = None, None
            dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
            # if Q_e:
            #     dQs = dQs.mean(0)
            # if p_e:
            #     dps = dps.mean(0)

            grads = (dQs, dps, dGs, dhs, dAs, dbs, dFs)

            return grads

    return LCPFunctionFn.apply
