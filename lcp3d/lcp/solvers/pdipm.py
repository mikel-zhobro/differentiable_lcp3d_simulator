"""Adapted from qpth: https://github.com/locuslab/qpth
"""

import torch

from ..util import efficient_btriunpack

from ..util import get_sizes

shown_btrifact_warning = False


# @profile
def lu_hack(x):
    global shown_btrifact_warning
    try:
        return x.lu(pivot=not x.is_cuda)
    except TypeError:
        if not shown_btrifact_warning:
            print(
                """----------
lcp warning: Pivoting will always happen and will significantly
slow down your code. Please use the master branch of PyTorch
to get a version that disables pivoting on the GPU.
----------
"""
            )
            shown_btrifact_warning = True
        return x.lu()


INACC_ERR = """
--------
lcp warning: Returning an inaccurate and potentially incorrect solution.

Some residual is large.
Your problem may be infeasible or difficult.

You can try using the verbose option to check the convergence status of
our solver while increasing the number of iterations.

Advanced users:
You can also try to enable iterative refinement in the solver:
https://github.com/locuslab/qpth/issues/6
--------
"""


# @profile
def forward(
    Q,
    p,
    G,
    h,
    A,
    b,
    F,
    Q_LU,
    S_LU,
    R,
    eps=1e-12,
    verbose=-1,
    not_improved_lim=3,
    max_iter=20,
):
    """
    Q_LU, S_LU, R = pre_factor_kkt(Q, G, A)
    """
    nineq, nz, neq, batch_size = get_sizes(G, A)

    # Find initial values
    d = Q.new_ones(batch_size, nineq)
    factor_kkt(S_LU, R, d)
    x, s, z, y = solve_kkt(
        Q_LU,
        d,
        G,
        A,
        S_LU,
        p,
        Q.new_zeros(batch_size, nineq),
        -h,
        -b if neq > 0 else None,
    )

    # Make all of the slack variables >= 1.
    M = torch.min(s, 1)[0]
    M = M.view(M.size(0), 1).repeat(1, nineq)
    I = M <= 0
    s[I] -= M[I] - 1

    # Make all of the inequality dual variables >= 1.
    M = torch.min(z, 1)[0]
    M = M.view(M.size(0), 1).repeat(1, nineq)
    I = M <= 0
    z[I] -= M[I] - 1

    best = {"resids": None, "x": None, "z": None, "s": None, "y": None}
    nNotImproved = 0

    for i in range(max_iter):
        # affine scaling direction
        rx = (
            (torch.bmm(y.unsqueeze(1), A).squeeze(1) if neq > 0 else 0.0)
            + torch.bmm(z.unsqueeze(1), G).squeeze(1)
            + torch.bmm(x.unsqueeze(1), Q.transpose(1, 2)).squeeze(1)
            + p
        )
        rs = z
        rz = (
            torch.bmm(x.unsqueeze(1), G.transpose(1, 2)).squeeze(1)
            + s
            - h
            - torch.bmm(z.unsqueeze(1), F.transpose(1, 2)).squeeze(1)
        )
        ry = (
            torch.bmm(x.unsqueeze(1), A.transpose(1, 2)).squeeze(1) - b
            if neq > 0
            else 0.0
        )
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq * mu

        d = z / s
        try:
            factor_kkt(S_LU, R, d)
        except:
            return best["x"], best["y"], best["z"], best["s"]

        if verbose > 0:
            print(
                "iter: {}, pri_resid: {:.5e}, dual_resid: {:.5e}, mu: {:.5e}".format(
                    i, pri_resid.mean(), dual_resid.mean(), mu.mean()
                )
            )
        if best["resids"] is None:
            best["resids"] = resids
            best["x"] = x.clone()
            best["z"] = z.clone()
            best["s"] = s.clone()
            best["y"] = y.clone() if y is not None else None
            nNotImproved = 0
        else:
            I = resids < best["resids"]
            if I.sum() > 0:
                nNotImproved = 0
            else:
                nNotImproved += 1
            I_nz = I.repeat(nz, 1).t()
            I_nineq = I.repeat(nineq, 1).t()
            # best['resids'][I] = resids[I]
            # best['x'][I_nz] = x[I_nz]
            # best['z'][I_nineq] = z[I_nineq]
            # best['s'][I_nineq] = s[I_nineq]
            best["resids"].masked_scatter_(I, resids[I])
            best["x"].masked_scatter_(I_nz, x[I_nz])
            best["z"].masked_scatter_(I_nineq, z[I_nineq])
            best["s"].masked_scatter_(I_nineq, s[I_nineq])
            if neq > 0:
                I_neq = I.repeat(neq, 1).t()
                best["y"][I_neq] = y[I_neq]
        if (
            nNotImproved == not_improved_lim
            or best["resids"].max().item() < eps
            or mu.min().item() > 1e100
        ):
            if best["resids"].max() > 1.0 and verbose >= 0:
                print(INACC_ERR)
            return best["x"], best["y"], best["z"], best["s"]

        dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

        # compute centering directions
        alpha = torch.min(
            torch.min(get_step(z, dz_aff), get_step(s, ds_aff)),
            torch.ones(batch_size).type_as(Q),
        )
        alpha_nineq = alpha.repeat(nineq, 1).t()
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4) ** 3

        rx = Q.new_zeros(batch_size, nz)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff) / s
        rz = Q.new_zeros(batch_size, nineq)
        ry = Q.new_zeros(batch_size, neq)

        dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry)

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = torch.min(
            0.999 * torch.min(get_step(z, dz), get_step(s, ds)), Q.new_ones(batch_size)
        )
        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()

        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None

    if best["resids"].max() > 1.0 and verbose >= 0:
        print(INACC_ERR)
        print(best["resids"].max())
    return best["x"], best["y"], best["z"], best["s"]


def get_step(v, dv):
    a = -v / dv
    a[dv > 0] = max(1.0, a.max())
    step = a.min(1)[0].squeeze()
    return step


def unpack_kkt(v, nz, nineq, neq):
    i = 0
    x = v[:, i : i + nz]
    i += nz
    s = v[:, i : i + nineq]
    i += nineq
    z = v[:, i : i + nineq]
    i += nineq
    y = v[:, i : i + neq]
    return x, s, z, y


def kkt_resid_reg(Q_tilde, D_tilde, G, A, F_tilde, eps, dx, ds, dz, dy, rx, rs, rz, ry):
    dx, ds, dz, dy = [
        x.unsqueeze(2) if x is not None else None for x in [dx, ds, dz, dy]
    ]
    resx = Q_tilde.bmm(dx) + G.transpose(1, 2).bmm(dz) + rx.unsqueeze(2)
    if dy is not None:
        resx += A.transpose(1, 2).bmm(dy)
    ress = D_tilde.bmm(ds) + dz + rs.unsqueeze(2)
    resz = G.bmm(dx) + ds + F_tilde.bmm(dz) + rz.unsqueeze(2)  # XXX
    resy = A.bmm(dx) - eps * dy + ry.unsqueeze(2) if dy is not None else None
    resx, ress, resz, resy = (
        v.squeeze(2) if v is not None else None for v in (resx, ress, resz, resy)
    )

    return resx, ress, resz, resy


def solve_kkt_ir(Q, D, G, A, F, rx, rs, rz, ry, niter=1):
    """Inefficient iterative refinement."""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    eps = 1e-7
    Q_tilde = Q + eps * torch.eye(nz).type_as(Q).repeat(nBatch, 1, 1)
    D_tilde = D + eps * torch.eye(nineq).type_as(Q).repeat(nBatch, 1, 1)

    # XXX Might not workd for batch size > 1
    C_tilde = -eps * torch.eye(neq + nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)
    if F is not None:  # XXX inverted sign for F below
        C_tilde[:, :nineq, :nineq] -= F
    F_tilde = C_tilde[:, :nineq, :nineq]

    dx, ds, dz, dy = factor_solve_kkt_reg(
        Q_tilde, D_tilde, G, A, C_tilde, rx, rs, rz, ry, eps
    )
    resx, ress, resz, resy = kkt_resid_reg(
        Q, D, G, A, F_tilde, eps, dx, ds, dz, dy, rx, rs, rz, ry
    )
    for k in range(niter):
        ddx, dds, ddz, ddy = factor_solve_kkt_reg(
            Q_tilde,
            D_tilde,
            G,
            A,
            C_tilde,
            -resx,
            -ress,
            -resz,
            -resy if resy is not None else None,
            eps,
        )
        dx, ds, dz, dy = [
            v + dv if v is not None else None
            for v, dv in zip((dx, ds, dz, dy), (ddx, dds, ddz, ddy))
        ]
        resx, ress, resz, resy = kkt_resid_reg(
            Q, D, G, A, F_tilde, eps, dx, ds, dz, dy, rx, rs, rz, ry
        )

    return dx, ds, dz, dy


def factor_solve_kkt_reg(Q_tilde, D, G, A, C_tilde, rx, rs, rz, ry, eps):
    nineq, nz, neq, nBatch = get_sizes(G, A)

    H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q_tilde)
    H_[:, :nz, :nz] = Q_tilde
    H_[:, -nineq:, -nineq:] = D
    if neq > 0:
        # H_ = torch.cat([torch.cat([Q, torch.zeros(nz,nineq).type_as(Q)], 1),
        # torch.cat([torch.zeros(nineq, nz).type_as(Q), D], 1)], 0)
        A_ = torch.cat(
            [
                torch.cat(
                    [G, torch.eye(nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)], 2
                ),
                torch.cat([A, torch.zeros(nBatch, neq, nineq).type_as(Q_tilde)], 2),
            ],
            1,
        )
        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)], 2)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    H_LU = lu_hack(H_)

    invH_A_ = A_.transpose(1, 2).lu_solve(*H_LU)  # H-1 AT
    invH_g_ = g_.lu_solve(*H_LU)  # H-1 g

    S_ = torch.bmm(A_, invH_A_)  # A H-1 AT
    # A H-1 AT + C_tilde
    S_ -= C_tilde
    S_LU = lu_hack(S_)
    # [(H-1 g)T AT]T - h = A H-1 g - h
    t_ = torch.bmm(invH_g_.unsqueeze(1), A_.transpose(1, 2)).squeeze(1) - h_
    # w = (A H-1 AT + C_tilde)-1 (A H-1 g - h) <= Av - eps I w = h
    w_ = -t_.lu_solve(*S_LU)
    # Shouldn't it be just g (no minus)?
    # (Doesn't seem to make a difference, though...)
    t_ = -g_ - w_.unsqueeze(1).bmm(A_).squeeze()  # -g - AT w
    v_ = t_.lu_solve(*H_LU)  # v = H-1 (-g - AT w)

    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def factor_solve_kkt(Q_tilde, D_tilde, A_, C_tilde, rx, rs, rz, ry, ns):
    nineq, nz, neq, nBatch = ns

    H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q_tilde)
    H_[:, :nz, :nz] = Q_tilde
    H_[:, -nineq:, -nineq:] = D_tilde
    if neq > 0:
        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    H_LU = lu_hack(H_)

    invH_A_ = A_.transpose(1, 2).lu_solve(*H_LU)
    invH_g_ = g_.lu_solve(*H_LU)

    S_ = torch.bmm(A_, invH_A_) + C_tilde
    S_LU = lu_hack(S_)
    t_ = torch.bmm(invH_g_.unsqueeze(1), A_.transpose(1, 2)).squeeze(1) - h_
    w_ = -t_.lu_solve(*S_LU)
    t_ = -g_ - w_.unsqueeze(1).bmm(A_).squeeze()
    v_ = t_.lu_solve(*H_LU)

    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """Solve KKT equations for the affine step"""

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]

    nineq, nz, neq, nBatch = get_sizes(G, A)

    invQ_rx = rx[:, :, None].lu_solve(*Q_LU)[:, :, 0]  # Q-1 rx
    if neq > 0:
        # A Q-1 rx - ry
        # G Q-1 rx + rs / d - rz
        h = torch.cat(
            [
                invQ_rx[:, None].bmm(A.transpose(1, 2))[:, 0] - ry,
                invQ_rx[:, None].bmm(G.transpose(1, 2))[:, 0] + rs / d - rz,
            ],
            1,
        )
    else:
        h = invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz

    w = -(h[:, :, None].lu_solve(*S_LU))[:, :, 0]  # S-1 h =

    g1 = -rx - w[:, None, neq:].bmm(G)[:, 0]  # -rx - GT w = -rx -GT S-1 h
    if neq > 0:
        g1 -= w[:, None, :neq].bmm(A)[:, 0]  # - AT w = -AT S-1 h
    g2 = -rs - w[:, neq:]

    dx = g1[:, :, None].lu_solve(*Q_LU)[:, :, 0]  # Q-1 g1 = - Q-1 AT S-1 h
    ds = g2 / d  # g2 / d = (-rs - w) / d
    dz = w[:, neq:]
    dy = w[:, :neq] if neq > 0 else None

    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, F, A):
    """Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    try:
        Q_LU = lu_hack(Q)
    except:
        raise RuntimeError(
            """
lcp Error: Cannot perform LU factorization on Q.
Please make sure that your Q matrix is PSD and has
a non-zero diagonal.
"""
        )

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    #
    # We compute a partial LU decomposition of the S matrix
    # that can be completed once D^{-1} is known.
    # See the 'Block LU factorization' part of our website
    # for more details.

    G_invQ_GT = torch.bmm(G, G.transpose(1, 2).lu_solve(*Q_LU)) + F
    R = G_invQ_GT.clone()
    S_LU_pivots = (
        torch.IntTensor(range(1, 1 + neq + nineq))
        .unsqueeze(0)
        .repeat(nBatch, 1)
        .type_as(Q)
        .int()
    )
    if neq > 0:
        invQ_AT = A.transpose(1, 2).lu_solve(*Q_LU)
        A_invQ_AT = torch.bmm(A, invQ_AT)
        G_invQ_AT = torch.bmm(G, invQ_AT)

        LU_A_invQ_AT = lu_hack(A_invQ_AT)
        P_A_invQ_AT, L_A_invQ_AT, U_A_invQ_AT = efficient_btriunpack(*LU_A_invQ_AT)
        P_A_invQ_AT = P_A_invQ_AT.type_as(A_invQ_AT)

        S_LU_11 = LU_A_invQ_AT[0]
        U_A_invQ_AT_inv = (P_A_invQ_AT.bmm(L_A_invQ_AT)).lu_solve(*LU_A_invQ_AT)
        S_LU_21 = G_invQ_AT.bmm(U_A_invQ_AT_inv)
        T = G_invQ_AT.transpose(1, 2).lu_solve(*LU_A_invQ_AT)
        S_LU_12 = U_A_invQ_AT.bmm(T)
        S_LU_22 = Q.new_zeros(nBatch, nineq, nineq)
        S_LU_data = torch.cat(
            (torch.cat((S_LU_11, S_LU_12), 2), torch.cat((S_LU_21, S_LU_22), 2)), 1
        )
        S_LU_pivots[:, :neq] = LU_A_invQ_AT[1]

        R -= G_invQ_AT.bmm(T)
    else:
        S_LU_data = Q.new_zeros(nBatch, nineq, nineq)

    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R


factor_kkt_eye = None

# @profile
def factor_kkt(S_LU, R, d):
    """Factor the U22 block that we can only do after we know D."""
    nBatch, nineq = d.size()
    neq = S_LU[1].size(1) - nineq
    # TODO There's probably a better way to add a batched diagonal.
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        # print('Updating batchedEye size.')
        factor_kkt_eye = torch.eye(nineq, dtype=torch.bool).repeat(nBatch, 1, 1)
    # T = R.clone()
    # T[factor_kkt_eye] += (1. / d).view(-1)
    # more efficient version of these two lines in pytorch versions > 0.3.1
    T = torch.zeros_like(R)
    T.masked_scatter_(factor_kkt_eye, (1.0 / d).view(-1))
    T += R.clone()

    T_LU = lu_hack(T)

    global shown_btrifact_warning
    if shown_btrifact_warning or not T.is_cuda:
        # TODO Don't use pivoting in most cases because
        # torch.btriunpack is inefficient here:
        oldPivotsPacked = S_LU[1][:, -nineq:] - neq
        oldPivots, _, _ = efficient_btriunpack(
            T_LU[0], oldPivotsPacked, unpack_data=False
        )
        newPivotsPacked = T_LU[1]
        newPivots, _, _ = efficient_btriunpack(
            T_LU[0], newPivotsPacked, unpack_data=False
        )

        # Re-pivot the S_LU_21 block.
        if neq > 0:
            S_LU_21 = S_LU[0][:, -nineq:, :neq]
            S_LU[0][:, -nineq:, :neq] = newPivots.transpose(1, 2).bmm(
                oldPivots.bmm(S_LU_21)
            )

        # Add the new S_LU_22 block pivots.
        S_LU[1][:, -nineq:] = newPivotsPacked + neq

    # Add the new S_LU_22 block.
    S_LU[0][:, -nineq:, -nineq:] = T_LU[0]
