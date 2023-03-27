from __future__ import annotations

import math
import time
from typing import Sequence, Union, TYPE_CHECKING

import torch
from pytorch3d.transforms import (
    so3_exp_map,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from . import contacts as contacts_module
from . import engines as engines_module

from .contacts import filter_contact_points
from .utils import get_instance, get_tensor, get_directions, J
from ..config import get_config

if TYPE_CHECKING:
    from .contacts import Contact
    from .bodies import RigidBody
    from .forces import F, MultiWrench
    from .constraints import Constraint as Constraint


default_config = get_config()

def trigger_update(func):
    """Trigger update_collision_visual_geometries when called."""
    def wrap(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.update_collision_visual_geometries()
        # what should be triggered from self?
        # write here..
        return ret
    return wrap

class World:
    """A physics simulation world, with bodies and constraints."""

    def __init__(
        self,
        bodies: list[RigidBody],
        constraints: Union[list[Constraint], None] = None,
        gravity = default_config.g,
        dt=default_config.dt,
        writer=None,
        log=False,
    ):
        self._base_tensor: torch.Tensor = (
            bodies[0]._base_tensor
            if len(bodies) > 0
            else get_tensor(0, base_tensor=None)
        )
        # State of the world
        self.t = 0
        self.bodies = bodies
        # -> init state
        self.p0 = torch.stack([b.init_p for b in self.bodies])                  # (N, 3)
        self.R0 = torch.stack([b.init_R for b in self.bodies])                  # (N, 3, 3)
        self.v_w0 = torch.stack([b.init_vw for b in self.bodies])              # (N, 6)
        self.gravity = self.p0.new_zeros((len(self.bodies), 6))             # (N, 6)
        self.gravity[:, 2] = gravity*torch.stack([b.mass for b in self.bodies])
        self.forces = self.p0.new_zeros((len(self.bodies), 6))              # (N, 6)
        # -> current state
        self.p, self.R, self.vw = self.p0.clone(), self.R0.clone(), self.v_w0.clone()

        # set indexes and world pointer
        [b.set_ix(i, self) for i, b in enumerate(self.bodies)]

        # Load classes from string name defined in utils
        self.engine: engines_module.Engine
        if default_config.engine == "QPEngine":
            self.engine = engines_module.QPEngine()
        else:
            raise NotImplementedError("Only qp engine is implemented")

        self.contact_detector: contacts_module.ContactHandler
        if default_config.contact_handler == "OdeContactHandler":
            self.contact_detector = contacts_module.OdeContactHandler(bodies)
        else:
            raise NotImplementedError("Only ode contact handler is implemented")

        self.writer = writer
        self.log = log

        # Simulation parameters
        self.dt = dt
        self.eps = default_config.eps
        self.tol = default_config.tol
        self.fric_dirs = default_config.fric_dirs
        self.post_stab = default_config.post_stabilization
        self.vec_len = 6

        # Constraints aka joints
        self.constraints: list[Constraint] = [] if constraints is None else constraints
        for ctr in self.constraints:
            ctr.set_ix(
                bodies.index(ctr.body1), bodies.index(ctr.body2) if ctr.body2 else -10
            )

        self.num_constraints = sum([ctr.num_constraints for ctr in self.constraints])

        # Make sure contacts are feasible(i.e. no penetration and co)
        self.contacts: list[Contact] = list()
        self.strict_no_pen = default_config.strict_no_penetration
        self.correct_initial_penetration = default_config.correct_initial_penetration
        self.init_contacts()

    def get_v_w_s(self):
        return self.vw

    def get_positions(self):
        return self.p

    def get_orients(self):
        return self.R

    def get_forces(self):
        return (self.gravity + self.forces).view(-1)

    def reset_forces(self):
        self.forces = self._base_tensor.new_zeros((len(self.bodies), 6))

    # @trigger_update
    def set_state(self, p, R, vw):
        self.p = p
        self.R = R
        self.vw = vw
        self.update_collision_visual_geometries()

    def update_collision_visual_geometries(self):
        """Update collision and visual geometries of all bodies."""
        [b.update_collision_geom() for b in self.bodies]

    def step(self, fixed_dt=True):
        """Simulates a step dt(or less in case of contact and fixed_dt=False) and resets the force inputs."""
        sub_steps = 0
        tmp_start = time.time()
        N_contacts = []

        dt = self.dt
        if fixed_dt:
            end_t = self.t + self.dt
            while self.t < end_t:
                dt = end_t - self.t
                sub_steps += self._step_dt(dt)
                N_contacts.append(len(self.contacts))
        else:
            sub_steps += self._step_dt(dt)
        # print(f"N substeps:{ sub_steps}, time: {time.time()-tmp_start}, Contacts: {N_contacts}")

        # Reset inputs
        self.reset_forces()

    def lcp(self, dt):
        v, F_ext = self.get_v_w_s().view(-1), self.get_forces()

        Je, Jc, Jf = self.Je(), self.Jc(), self.Jf()

        M, Coriolis = self.M(), self.Coriolis()
        E, mu, restitutions =  self.E(), self.mu(), self.restitutions()

        new_v = self.engine.solve_dynamics(
            dt, F_ext, v, Je, Jc, Jf, M, E, Coriolis, mu, restitutions
        )  # here the force is integrated
        return new_v

    def _step_dt(self, dt):
        """Simulates a step dt(or less in case of contact)"""
        start_v = self.get_v_w_s().clone()
        start_R = self.get_orients().clone()
        start_p = self.get_positions().clone()
        start_contacts = [contact for contact in self.contacts]

        assert start_p.shape[-1] == 3 and start_R.shape[-1] == 3
        assert start_v.shape[-1] == 6

        sub_steps = 0
        # A. Solve LCP
        while True:
            # 1. Compute new velocities and integrate them
            new_vw = self.lcp(dt).view(-1, self.vec_len)  # here the force is integrated
            new_p, new_R = self.integrate_velocities(dt, self.p, self.R, new_vw)
            self.set_state(new_p, new_R, new_vw)

            # 2. Check whether all penetrations are > tolerance
            #    if not: halve dt and reset poses, velocities and contacts of the bodies.
            self.update_contacts()

            step_too_small = (
                not self.strict_no_pen
                or dt < self.dt / 2**default_config.num_sub_steps
            )
            not_penetrated = all(
                [c.penetration.item() <= self.tol for c in self.contacts]
            )
            if step_too_small or not_penetrated:
                break
            else:
                dt /= 2
                self.R = start_R.clone()
                self.p = start_p.clone()
                self.vw = start_v.clone()
                self.contacts = [contact for contact in start_contacts]
                sub_steps += 1

        # B. Post stabilization #TODO
        if self.post_stab:
            self.post_stabilization(dt)

        self.t += dt

        for b in self.bodies:
            b.visualize()
        return sub_steps + 1

    def add_force(self, f, multiforce: MultiWrench):
        # used in sphere_with_initialized_state.py, lcp_3d_wrapper.py
        ix = multiforce.ix
        rot = self.R[ix]
        my_f  = multiforce.get_forward_matrix(rot) @ f

        assert my_f.numel() == 6, "make sure external forces has size of 6. forces+torques"
        self.forces[ix] += my_f

        # if self.visualize_external_forces:
        n_f = multiforce.n_force
        vis_f = f[:3*n_f].view(-1, 3).detach().cpu().numpy()
        vis_t = f[3*n_f:].view(-1, 3).detach().cpu().numpy()
        [self.bodies[ix].visualize_force(f, multiforce.f_poses[i], i) for i, f in enumerate(vis_f)]
        [self.bodies[ix].visualize_torque(t, multiforce.t_poses[i], n_f+i) for i, t in enumerate(vis_t)]

    def integrate_velocities(self, dt, p, R, vw):
        # used in world.py
        newR = torch.einsum('bij,bjk->bik', so3_exp_map(vw[:, 3:] * dt), R)
        newp = vw[:, :3] * dt + p
        # delta = se3_exp_map(self._v_w[None, ...] * dt).squeeze().T
        # newR = delta[:3,:3] @ self.rot
        # newp = delta[:3,-1] + self._p
        # check_R(newR)
        # print(f"w={self.w.view(-1)}")
        return newp, newR

    def post_stabilization(self, dt):
        print("Post stabilization")
        v_tmp, Je, Jc, M, restitutions = (
            self.get_v_w_s().view(-1),
            self.Je(),
            self.Jc(),
            self.M(),
            self.restitutions(),
        )
        dp = self.engine.post_stabilization(v_tmp, M, Je, Jc, restitutions).squeeze(0)
        dp /= 2  # XXX Why 1/2 factor?
        # XXX Clean up / Simplify this update?
        new_vw = dp.view(-1, self.vec_len)
        new_p, new_R = self.integrate_velocities(dt, self.p, self.R, new_vw)
        self.set_state(new_p, new_R, new_vw)

        self.update_contacts()  # XXX Necessary to recheck contacts?

    def Je(self):
        # Depends only on self.constraints
        Je = self._base_tensor.new_zeros(
            self.num_constraints, self.vec_len * len(self.bodies)
        )
        row = 0
        for joint in self.constraints:
            J1, J2 = joint.J()
            i1 = joint.ix1
            i2 = joint.ix2
            Je[row : row + J1.size(0), i1 * self.vec_len : (i1 + 1) * self.vec_len] = J1
            if J2 is not None:
                Je[
                    row : row + J2.size(0), i2 * self.vec_len : (i2 + 1) * self.vec_len
                ] = J2
            row += J1.size(0)
        return Je

    def Jc(self):
        # Depends on self.contacts
        Jc = self._base_tensor.new_zeros(
            len(self.contacts), self.vec_len * len(self.bodies)
        )
        # print(Jc.shape)
        for i, c in enumerate(self.contacts):
            # J1 = torch.cat([torch.cross(c.xa, c.normal), c.normal])
            # J2 = -torch.cat([torch.cross(c.xb, c.normal), c.normal])
            xa, xb = c.xa, c.xb # for the case where contact points are not differentiable XXX
            # xa, xb = c.point_a_w - self.p[c.index1], c.point_b_w - self.p[c.index2]
            J1 = torch.cat([c.normal, torch.cross(xa, c.normal)])
            J2 = -torch.cat([c.normal, torch.cross(xb, c.normal)])
            Jc[i, c.index1 * self.vec_len : (c.index1 + 1) * self.vec_len] = J1
            Jc[i, c.index2 * self.vec_len : (c.index2 + 1) * self.vec_len] = J2
        return Jc

    def Jf(self):
        # Depends on self.contacts
        Jf = self._base_tensor.new_zeros(
            len(self.contacts) * self.fric_dirs, self.vec_len * len(self.bodies)
        )
        for i, c in enumerate(self.contacts):
            directions = get_directions(
                c.d1, c.d2, depth=int(math.log(self.fric_dirs, 2))
            )
            xa, xb = c.xa, c.xb # for the case where contact points are not differentiable XXX
            # xa, xb = c.point_a_w - self.p[c.index1], c.point_b_w - self.p[c.index2]
            J1 = J(xa, *directions)
            J2 = J(xb, *directions)

            Jf[
                i * self.fric_dirs : (i + 1) * self.fric_dirs,
                c.index1 * self.vec_len : (c.index1 + 1) * self.vec_len,
            ] = J1
            Jf[
                i * self.fric_dirs : (i + 1) * self.fric_dirs,
                c.index2 * self.vec_len : (c.index2 + 1) * self.vec_len,
            ] = -J2
        return Jf

    # --------------------------------------------------------------------------------------------
    # Mass matrix and Coriolis, can be parallelized TODO
    def M(self):
        return torch.block_diag(*[b.M_world for b in self.bodies])

    def Coriolis(self):
        return torch.cat(
            [
                torch.cat([self._base_tensor.new_zeros(3), b.Coriolis])
                for b in self.bodies
            ]
        )
    # --------------------------------------------------------------------------------------------

    def restitutions(self):
        restitutions = self._base_tensor.new_empty(len(self.contacts))
        for i, c in enumerate(self.contacts):
            r1 = self.bodies[c.index1].restitution
            r2 = self.bodies[c.index2].restitution
            restitutions[i] = r1 * r2
        # restitutions[i] = math.sqrt(r1 * r2)
        return restitutions

    def mu(self):
        mu = self._base_tensor.new_zeros(len(self.contacts))
        for i, c in enumerate(self.contacts):
            mu[i] = self.bodies[c.index1].fric_coeff * self.bodies[c.index2].fric_coeff
        return torch.diag(mu)

    def E(self):
        # depends on contacts
        def _memoized_E(num_contacts):
            n = self.fric_dirs * num_contacts
            E = self._base_tensor.new_zeros(n, num_contacts)
            for i in range(num_contacts):
                E[i * self.fric_dirs : (i + 1) * self.fric_dirs, i] += 1
            return E

        return _memoized_E(len(self.contacts))

    def update_contacts(self):
        self.contacts.clear()
        t_start = time.time()
        # ODE contact detection
        self.contacts = self.contact_detector.update_contacts(self.p, self.R)

        if default_config.filter_contacts:
            filter_contact_points(self.contacts)
        if self.log and self.writer is not None:
            self.writer(
                label="Time_Analysis/Contact_detection_time", data=time.time() - t_start
            )

    def init_contacts(self):
        def update_body_poses(pairs: dict):
            for (geom1, geom2), (normal, pen) in pairs.items():
                ix1, ix2 = self.bodies[geom1].ix, self.bodies[geom2].ix
                if geom1 == 0:
                    self.p[ix2] = self.p[ix2] - normal * torch.abs(pen)
                    self.bodies[geom2].visualize()
                elif geom2 == 0:
                    self.p[ix1] = self.p[ix1] + normal * torch.abs(pen)
                    self.bodies[geom1].visualize()
                else:  # Half plane projection
                    self.p[ix1] = self.p[ix1] + 0.5 * normal * torch.abs(pen)
                    self.p[ix2] = self.p[ix2] - 0.5 * normal * torch.abs(pen)
                    self.bodies[geom1].visualize()
                    self.bodies[geom2].visualize()

        self.update_contacts()
        if not all([c.penetration.item() <= self.tol for c in self.contacts]):
            if self.strict_no_pen and not self.correct_initial_penetration:
                raise AssertionError("Interpenetration detected at start:\n")
            elif self.correct_initial_penetration:
                initial_penetration = True
                while initial_penetration:
                    print("Correcting penetration")
                    # Get maximum penetration and contact normal between two objects
                    pairs = {}
                    for c in self.contacts:
                        if (c.index1, c.index2) in pairs.keys():
                            if (
                                torch.abs(c.penetration)
                                > pairs[(c.index1, c.index2)][1]
                            ):
                                pairs[(c.index1, c.index2)] = [c.normal, c.penetration]
                        elif (c.index2, c.index1) in pairs.keys():
                            if (
                                torch.abs(c.penetration)
                                > pairs[(c.index2, c.index1)][1]
                            ):
                                pairs[(c.index2, c.index1)] = [c.normal, c.penetration]
                        else:
                            pairs[(c.index1, c.index2)] = [c.normal, c.penetration]

                    # Perform half plane projection on all the bodies except ground plane
                    update_body_poses(pairs)

                    self.update_collision_visual_geometries()
                    self.update_contacts()
                    if all([c.penetration.item() <= self.tol for c in self.contacts]):
                        initial_penetration = False
