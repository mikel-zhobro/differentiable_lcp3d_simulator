import torch
import numpy as np
from typing import Union

def get_skew(r: torch.Tensor): # same as hat
    # Computes skew matrix of shape (N, 3 ,3) from a vector r of shape (N, 3)
    my_skew = r.new_zeros((r.shape[0], 3, 3))
    my_skew[:, 0, 1] = -r[:, 2]
    my_skew[:, 1, 0] =  r[:, 2]
    my_skew[:, 0, 2] =  r[:, 1]
    my_skew[:, 2, 0] = -r[:, 1]
    my_skew[:, 1, 2] = -r[:, 0]
    my_skew[:, 2, 1] =  r[:, 0]
    return my_skew


class F:
    TForce = 0
    TTorque = 1
    TWrench = 2
    NumType = [3, 3, 6]
    def __init__(self, f_type: int, pos: list[float]=[0.,0.,0.], ctrl_min=-1e6, ctrl_max=1e6) -> None:
        assert len(pos) == 3, "pos should be a [x,y,z] position list."
        self.f_type = f_type
        self.pos = pos
        self.num = F.NumType[f_type]
        self.ctrl_min = self._get_limit(ctrl_min)
        self.ctrl_max = self._get_limit(ctrl_max)


    def _get_limit(self, limit):
        # if it is a number, then we return a vector of that number
        if isinstance(limit, float):
            _limit = limit * np.ones(self.num)
        elif isinstance(limit, (list, np.ndarray)):
            if len(limit) == 2 and self.f_type == F.TWrench:
                limit = [limit[0], limit[0], limit[0], limit[1], limit[1], limit[1]]
            assert len(limit) == self.num, f"limit should have {self.num} elements or 2 if is a wrench."
            _limit = limit
        else:
            raise ValueError(f"limit should be a float or a list of length {self.num}")
        return _limit

    @staticmethod
    def from_force(x: torch.Tensor):
        return torch.cat([x, x.new_zeros(3)])
    @staticmethod
    def from_torque(x: torch.Tensor):
        return torch.cat([x.new_zeros(3), x])
    @staticmethod
    def from_wrench(x: torch.Tensor):
        return x

    def get_wrench(self, x: torch.Tensor):
        # Right now we compute the wrenches one by one.
        # We could also create slice_indices for Forces, Torques and Wrenches
        # and compute the wrenches in batch.
        assert x.dim()==1, f"Make sure the F/T/W has only 1 dimension"
        assert x.numel() in (3,6), f"Forces/Torques/Wrench can have a size of either 3 or 6 not {x.numel()}"

        return F.func_get_wrench[self.f_type](x)

    # smth we need
    func_get_wrench = [from_force.__func__,
                       from_torque.__func__,
                       from_wrench.__func__]


class MultiWrench:
    def __init__(self, ix: int, f_t_w: list[F]) -> None:
        # The forces/torques are ordered as follows in input vector: [F1, F2,.., FN, T1, T2,..,TK]
        self.ix = ix # index of the body in world.bodies
        self.types = [f.f_type for f in f_t_w]
        self.f_t_w = f_t_w
        self.forces = [f for f in f_t_w if f.f_type in [F.TForce, F.TWrench]]
        self.torques = [f for f in f_t_w if f.f_type in [F.TTorque, F.TWrench]]

        self.forces_bool = [int(f.f_type in [F.TForce, F.TWrench]) for f in f_t_w]
        self.torques_bool = [int(f.f_type in [F.TTorque, F.TWrench]) for f in f_t_w]

        self.nums = [f.num for f in self.forces]
        self.start_indexes = np.cumsum([0] + self.nums)

        self.n_force = len(self.forces)
        self.n_torque = len(self.torques)
        self.num = 3*self.n_force + 3*self.n_torque
        self.ctrl_min = np.concatenate([f.ctrl_min[:3] for f in self.forces] + [t.ctrl_min if t.f_type ==F.TTorque else t.ctrl_min[3:]  for t in self.torques])
        self.ctrl_max = np.concatenate([f.ctrl_max[:3] for f in self.forces] + [t.ctrl_max if t.f_type ==F.TTorque else t.ctrl_max[3:]  for t in self.torques])

    @property
    def f_poses(self):
        return [f.pos for f in self.forces]

    @property
    def t_poses(self):
        return [t.pos for t in self.torques]

    def get_absolute_positions(self, R: torch.Tensor, poses):
        return R @ R.new_tensor(poses).view(-1,3).T # shape: 3, N

    def get_forces(self, force_vec: torch.Tensor):
        return force_vec[:self.n_force*3]

    def get_torques(self, force_vec: torch.Tensor):
        return force_vec[self.n_force*3:]

    def get_forward_matrix(self, R: torch.Tensor):
        # F_cm = M [F1, F2,.., FN, T1, T2,..,TK]^T
        # where  M = [ [  J_F,    0]
        #              [J_F*_R_, J_T] ]
        abs_rel_positions = self.get_absolute_positions(R, self.f_poses).T # N x 3
        _r_=  get_skew(abs_rel_positions) # N x 3 x 3
        J_F = self.J_F(R)
        J_T = self.J_T(R)

        M = torch.block_diag(J_F, J_T) # constant, doesnt change
        M[J_F.shape[0]:, :J_F.shape[1]] = J_F @ torch.block_diag(*_r_) # depends on the positions
        return M

    def J_F(self, base_tensor: torch.Tensor):
        return base_tensor.new_ones(3).diag().repeat(1, self.n_force)

    def J_T(self, base_tensor: torch.Tensor):
        return base_tensor.new_ones(3).diag().repeat(1, self.n_torque)


def yield_body_multiforce(input_vec: Union[torch.Tensor, np.ndarray], ixs_controlled: list[int], nums: list[int],
                          list_forces_bool: list[list[int]], list_torques_bool: list[list[int]]):
    start = 0
    for ix, num, forces_bool, torques_bool in zip(ixs_controlled, nums, list_forces_bool, list_torques_bool):
        end = start + num
        in_vec = input_vec[..., start:end]

        n_force = sum(forces_bool) * 3
        forces_cumsum = 3 * np.cumsum(forces_bool)
        torques_cumsum = 3 * np.cumsum(torques_bool)
        force_list = np.split(in_vec[..., :n_force], forces_cumsum[:-1], axis=-1)
        torque_list = np.split(in_vec[..., n_force:], torques_cumsum[:-1], axis=-1)

        start = end
        yield ix, force_list, torque_list

def get_ix_force_torque_list(input_vec: Union[torch.Tensor, np.ndarray], ixs_controlled: list[int], nums: list[int],
                             list_forces_bool: list[list[int]], list_torques_bool: list[list[int]]):
    return list(yield_body_multiforce(input_vec, ixs_controlled, nums, list_forces_bool, list_torques_bool))