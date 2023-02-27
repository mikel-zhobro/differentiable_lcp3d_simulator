import torch
from typing import Union, Tuple
from abc import abstractmethod, ABC
from .utils import Indices
from .bodies import RigidBody

X = Indices.X
Y = Indices.Y


class Constraint(ABC):
    def __init__(self, body1: RigidBody, body2: Union[RigidBody, None] = None):
        self.static = True
        self.body1 = body1
        self.base_tensor = body1._base_tensor
        self.body2 = body2

        self.ix1: int
        self.ix2: int
        self.num_constraints: int

    @abstractmethod
    def J(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        pass

    def set_ix(self, ix1: int, ix2: int):
        self.ix1 = ix1
        self.ix2 = ix2


class XConstraint(Constraint):
    """Prevents motion in the X axis."""

    def __init__(self, body1: RigidBody):
        super().__init__(body1)
        self.num_constraints = 1

    def J(self):
        J = self.base_tensor.new_tensor([1, 0, 0, 0, 0, 0]).unsqueeze(0)
        return J, None


class YConstraint(Constraint):
    """Prevents motion in the Y axis."""

    def __init__(self, body1: RigidBody):
        super().__init__(body1)
        self.num_constraints = 1

    def J(self):
        J = self.base_tensor.new_tensor([0, 1, 0, 0, 0, 0]).unsqueeze(0)
        return J, None


class ZConstraint(Constraint):
    """Prevents motion in the Y axis."""

    def __init__(self, body1: RigidBody):
        super().__init__(body1)
        self.num_constraints = 1

    def J(self):
        J = self.base_tensor.new_tensor([0, 0, 1, 0, 0, 0]).unsqueeze(0)
        return J, None


class RotConstraint(Constraint):
    """Prevents Rotations"""

    def __init__(self, body1: RigidBody):
        super().__init__(body1)
        self.num_constraints = 1

    def J(self):
        J = self.base_tensor.new_tensor([0, 0, 0, 1, 1, 1]).unsqueeze(0)
        return J, None


class TotalConstraint(Constraint):
    """Prevents all motion."""

    def __init__(self, body1: RigidBody):
        super().__init__(body1)
        self.num_constraints = 6
        self.eye = torch.eye(self.num_constraints).type_as(self.base_tensor)
        body1.is_total_constraint = True

    def J(self):
        J = self.eye
        return J, None
