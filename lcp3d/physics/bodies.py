from abc import abstractmethod, ABC
from typing import Union, TYPE_CHECKING
import ode
import pybullet as pb
import torch
import numpy as np
from pytorch3d.transforms import (
    so3_exp_map,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from .utils import Indices, get_tensor, get_quat_xyzw, get_quat_xyzw_numpy
from .utils import process_vhacd_meshes, get_mesh_params, dist_from_bottom
from .utils import rotation, pose, velocity
from ..config import get_config

if TYPE_CHECKING:
    from .world import World

X = Indices.X
Y = Indices.Y

config = get_config()


class RigidBody(ABC):
    """
    Base class for rigid bodies.
    Rigid body properties like mass, inertia, coefficient of friction and restitution are defined.
    Everything is represented in world frame beside mass inertia
    - external forces: world frame
    - velocities:      world frame
    - mass inertia:    body frame (so that it stays constant)
    """

    def __init__(
        self,
        pos,
        vel=(0.0, 0.0, 0.0),
        mass=1.0,
        restitution=config.restitution,
        fric_coeff=config.fric_coeff,
        eps=config.eps,
        color=(1, 0, 0, 1),
        dim=config.dim,
        client_id=0,
        visualize_external_forces=False,
        pb_collision=False
    ):
        self.world: World
        self.ix: int = -1  # index in world bodies list
        # get base tensor to define dtype, device and layout for others
        self._base_tensor: torch.Tensor
        self._set_base_tensor(locals().values())

        self.eps = get_tensor(eps, base_tensor=self._base_tensor)
        self.dim = dim

        # Used only for initialization of collision and visual shapes
        p_tmp = pose(pos, base_tensor=self._base_tensor).view(-1)
        self.init_p = p_tmp[:3]
        self.init_vw = velocity(vel, base_tensor=self._base_tensor).view(-1)
        self.init_R = quaternion_to_matrix(p_tmp[3:])


        # Inertial parameters
        self.mass = mass
        self.fric_coeff = fric_coeff
        self.restitution = restitution

        self.gravity_force = (
            get_tensor([0, 0, config.g, 0, 0, 0], base_tensor=self._base_tensor)
            * self.mass
        )
        self.total_external_force = get_tensor(
            [0, 0, 0, 0, 0, 0], base_tensor=self._base_tensor
        )

        self.color = color
        self.client_id = client_id
        self.visualize_external_forces = visualize_external_forces
        self.debug_lines = {}

        self.is_total_constraint = False

        # prepare ode and pybullet
        self.geom_bullet = self._create_visual_geom(pb_collision=pb_collision)
        self._init_visual_geom()
        self.geom_ode = self._create_collision_geom()
        self._init_collision_geom()  # ODE

    def set_ix(self, i: int, world):
        self.ix = i
        self.world = world

    def set_mass(self, m):
        self.mass = m

    def set_fric_coeff(self, mu):
        self.fric_coeff = mu

    def set_restitution(self, c):
        self.restitution = c

    # Getters
    @property
    def rot(self):
        return self.world.R[self.ix]

    @property
    def position(self):
        return self.world.p[self.ix]
    @property
    def vw(self):
        return self.world.vw[self.ix]

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, m):
        self._mass = get_tensor(m, self._base_tensor)

    # --------------------------------------------------------------------------------------------
    # Move to world.py and parallelize TODO
    @property
    def M_world(self) -> torch.Tensor:
        return torch.block_diag(
            self.mass * torch.diag(self.mass.new_ones(self.dim)),
            self.rot @ self._get_ang_inertia(self.mass) @ self.rot.T,
        )

    @property
    def ang_inert_world(self):
        return self.rot @ self._get_ang_inertia(self.mass) @ self.rot.T

    @property
    def Coriolis(self):
        return torch.cross(
            self.vw[3:], self.rot @ self._get_ang_inertia(self.mass) @ self.rot.T @ self.vw[3:]
        )
    # --------------------------------------------------------------------------------------------

    @property
    def fric_coeff(self):
        return self._fric_coeff

    @fric_coeff.setter
    def fric_coeff(self, mu):
        self._fric_coeff = get_tensor(mu, self._base_tensor)

    @property
    def restitution(self):
        return self._restitution

    @restitution.setter
    def restitution(self, c):
        self._restitution = get_tensor(c, self._base_tensor)

    # ode and bullet
    def get_pose(self):
        return np.concatenate((self._numpy_position, self._numpy_orientation_xyzw))
    def get_vel(self):
        return self.vw.detach().cpu().numpy()

    @property
    def _numpy_position(self):
        # return self.world.p_numpy[self.ix]
        return self.world.p[self.ix].detach().cpu().numpy()

    @property
    def _numpy_rot(self):
        # return self.world.R_numpy[self.ix]
        return self.world.R[self.ix].detach().cpu().numpy()

    @property
    def _numpy_orientation(self):
        # return matrix_to_quaternion(torch.tensor(self._numpy_rot)).numpy()
        return matrix_to_quaternion(self.world.R[self.ix].clone().detach())

    @property
    def _numpy_orientation_xyzw(self):
        q = self._numpy_orientation
        return np.concatenate((q[1:], q[:1]), axis=-1)

    def visualize(self):
        pb.resetBasePositionAndOrientation(
            self.get_geom_bullet(),
            posObj=self._numpy_position,
            ornObj=self._numpy_orientation_xyzw,
            physicsClientId=self.client_id,
        )

    def visualize_force(self, force, pos, ix):
        self.debug_lines[ix] = pb.addUserDebugLine(
            parentObjectUniqueId=self.get_geom_bullet(),
            lineFromXYZ=pos,
            lineToXYZ=pos+force,
            lineColorRGB=[0, 0, 0],
            lineWidth=3,
            replaceItemUniqueId=self.debug_lines[ix] if ix in self.debug_lines.keys() else -1,
        )  # type: ignore

    def visualize_torque(self, torque, pos, ix):
        for i in range(3):
            toxyz = np.zeros(3)
            toxyz[i] = torque[i]
            rgb = [0] * 3
            rgb[i] = 1
            self.debug_lines[ix] = pb.addUserDebugLine(
                parentObjectUniqueId=self.get_geom_bullet(),
                lineFromXYZ=pos,
                lineToXYZ=pos+toxyz,  # [0,0,external_force[-1]],
                lineColorRGB=rgb,  # [0,0,0],
                lineWidth=3,
                replaceItemUniqueId=self.debug_lines[ix] if ix in self.debug_lines.keys() else -1,
            )  # type: ignore

    # update ode bodies
    def update_collision_geom(self):
        self.update_collision_body_position()
        self.update_collision_body_orientation()

    def update_collision_body_position(self):
        self.get_collision_geom().setPosition(self._numpy_position)  # type: ignore

    def update_collision_body_orientation(self):
        self.get_collision_geom().setQuaternion(self._numpy_orientation)  # type: ignore

    # Secondary
    def _set_base_tensor(self, args):
        """Check if any tensor provided and if so set as base tensor to
        use as base for other tensors' dtype, device and layout.
        """
        if hasattr(self, "_base_tensor") and self._base_tensor is not None:
            return

        for arg in args:
            if isinstance(arg, torch.Tensor):
                self._base_tensor = arg.clone().detach()
                return

        # if no tensor provided, use defaults
        self._base_tensor = get_tensor(0, base_tensor=None)

    # Virtual methods (to be implemented)
    def _init_collision_geom(self):
        self.geom_ode.setPosition(self.init_p.numpy())
        self.geom_ode.setQuaternion(matrix_to_quaternion(self.init_R).numpy())
        self.geom_ode.no_contact = set()

    @abstractmethod
    def _create_collision_geom(self):
        raise NotImplementedError

    @abstractmethod
    def _create_visual_geom(self, colID, visID, pb_collision: bool) -> int:
        geom_bullet = pb.createMultiBody(
            self.mass.item() if pb_collision else 0, # ghostify
            baseCollisionShapeIndex=colID,
            baseVisualShapeIndex=visID,
            physicsClientId=self.client_id,
        )  # type: ignore
        return geom_bullet

    def _init_visual_geom(self):
        pb.changeDynamics(
            self.get_geom_bullet(),
            -1,
            lateralFriction=self.fric_coeff,
            restitution=self.restitution,
            linearDamping=0.0,
            angularDamping=0.0,
            jointDamping =0.0,
            physicsClientId=self.client_id,
        )  # type: ignore
        quat_xyzw = get_quat_xyzw_numpy(matrix_to_quaternion(self.init_R).numpy())
        pb.resetBasePositionAndOrientation(
            self.geom_bullet, self.init_p.numpy(), quat_xyzw, physicsClientId=self.client_id  # type: ignore
        )
        pb.resetBaseVelocity(
            self.geom_bullet, self.init_vw[:3], self.init_vw[3:], physicsClientId=self.client_id  # type: ignore # XXX
        )

    @abstractmethod
    def _get_ang_inertia(self, mass):
        raise NotImplementedError

    def get_collision_geom(self):
        return self.geom_ode

    def get_geom_bullet(self) -> int:
        return self.geom_bullet


class Sphere(RigidBody):
    def __init__(
        self,
        pos,
        radius=1.0,
        vel=(0, 0, 0),
        mass=1.0,
        restitution=config.restitution,
        fric_coeff=config.fric_coeff,
        eps=config.eps,
        color=(1, 0, 0, 1),
        dim=config.dim,
        client_id=0,
        visualize_external_forces=False,
        pb_collision=False
    ):
        self._set_base_tensor(locals().values())
        self.rad: torch.Tensor = get_tensor(radius, base_tensor=self._base_tensor)
        super().__init__(
            pos,
            vel=vel,
            mass=mass,
            restitution=restitution,
            fric_coeff=fric_coeff,
            eps=eps,
            color=color,
            client_id=client_id,
            dim=dim,
            visualize_external_forces=visualize_external_forces,
            pb_collision=pb_collision
        )

    def _get_ang_inertia(self, mass):
        I = (
            torch.eye(self.dim).type(config.dtype).to(config.lcp_device)
            * 0.4
            * mass
            * self.rad**2
        )
        return I

    def _create_collision_geom(self):
        # Create ODE objects
        return ode.GeomSphere(None, self.rad.item() + 2 * self.eps)

    def _create_visual_geom(self, pb_collision=False):
        # Create PyBullet objects
        colSphereID = -1
        if pb_collision:
            colSphereID: int = pb.createCollisionShape(
                pb.GEOM_SPHERE, radius=self.rad.item(), physicsClientId=self.client_id
            )  # type: ignore
        visSphereID: int = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=self.rad.item(),
            rgbaColor=self.color,
            physicsClientId=self.client_id,
        )  # type: ignore
        return super()._create_visual_geom(colSphereID, visSphereID, pb_collision)


class Box(RigidBody):
    def __init__(
        self,
        pos,
        length=1.0,
        width=1.0,
        height=1.0,
        vel=(0.0, 0.0, 0.0),
        mass=1.0,
        restitution=config.restitution,
        fric_coeff=config.fric_coeff,
        eps=config.eps,
        color=(0.9, 0.8, 0.7, 1),
        dim=config.dim,
        client_id=0,
        visualize_external_forces=False,
        pb_collision=False
    ):
        self._set_base_tensor(locals().values())
        self.length = get_tensor(length, base_tensor=self._base_tensor)
        self.width = get_tensor(width, base_tensor=self._base_tensor)
        self.height = get_tensor(height, base_tensor=self._base_tensor)
        super().__init__(
            pos,
            vel=vel,
            mass=mass,
            restitution=restitution,
            fric_coeff=fric_coeff,
            eps=eps,
            color=color,
            client_id=client_id,
            dim=dim,
            visualize_external_forces=visualize_external_forces,
            pb_collision=pb_collision
        )

    def _get_ang_inertia(self, mass):
        Ixx = mass * (self.width**2 + self.height**2) / 12
        Iyy = mass * (self.length**2 + self.height**2) / 12
        Izz = mass * (self.width**2 + self.length**2) / 12
        return torch.diag(torch.stack([Ixx, Iyy, Izz]).view(-1))

    def _create_collision_geom(self):
        # Create ODE objects
        return ode.GeomBox(
            None,
            [
                self.length.item() + 2 * self.eps,
                self.width.item() + 2 * self.eps,
                self.height.item() + 2 * self.eps,
            ],
        )

    def _create_visual_geom(self, pb_collision=False):
        # Create PyBullet objects
        colBoxID = -1
        if pb_collision:
            colBoxID: int = pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[
                    self.length.item() / 2,
                    self.width.item() / 2,
                    self.height.item() / 2,
                ],
                physicsClientId=self.client_id,
            )  # type: ignore

        visBoxID: int = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[
                self.length.item() / 2,
                self.width.item() / 2,
                self.height.item() / 2,
            ],
            rgbaColor=self.color,
            physicsClientId=self.client_id,
        )  # type: ignore
        return super()._create_visual_geom(colBoxID, visBoxID, pb_collision)


class Mesh(RigidBody):
    def __init__(
        self,
        obj_file: str,
        pos,
        vel=(0, 0, 0),
        mass=1.0,
        restitution=config.restitution,
        fric_coeff=config.fric_coeff,
        eps=config.eps,
        color=(0.9, 0.8, 0.7, 1),
        dim=config.dim,
        vhacd=False,
        on_ground_plane=False,
        vis_obj_file: Union[str, None] = None,
        client_id=0,
        visualize_external_forces=False,
        pb_collision=False
    ):
        self._set_base_tensor(locals().values())
        self.obj_file = obj_file
        self.vis_obj_file = vis_obj_file

        if vhacd:
            self.vertices, self.faces = process_vhacd_meshes(self.obj_file)
        else:
            self.vertices, self.faces = get_mesh_params(self.obj_file)

        if on_ground_plane:
            pos[-1] = dist_from_bottom(self.obj_file)

        super().__init__(
            pos,
            vel=vel,
            mass=mass,
            restitution=restitution,
            fric_coeff=fric_coeff,
            eps=eps,
            color=color,
            client_id=client_id,
            dim=dim,
            visualize_external_forces=visualize_external_forces,
            pb_collision=pb_collision
        )
        self.prev_mass = -1

    def _get_ang_inertia(self, mass):
        # TODO: Use mesh vertices to calculate angular inertia
        diag_inertia = pb.getDynamicsInfo(
            self.get_geom_bullet(), -1, physicsClientId=self.client_id
        )[2]
        _m = mass.clone().detach()
        diag_inertia = [mass * (i / _m) for i in diag_inertia]
        return torch.diag(get_tensor(diag_inertia, self._base_tensor))

    def _create_collision_geom(self):
        # Create ODE objects
        mesh = ode.TriMeshData()
        # print(len(self.vertices))
        mesh.build(self.vertices, self.faces)
        return ode.GeomTriMesh(mesh)

    def _create_visual_geom(self, pb_collision=False):
        # Create PyBullet objects
        colMeshID = -1
        if pb_collision:
            colMeshID = pb.createCollisionShape(
                pb.GEOM_MESH,
                fileName=self.obj_file,
                physicsClientId=self.client_id,
            )  # type: ignore

        visMeshID = pb.createVisualShape(
            pb.GEOM_MESH,
            fileName=self.vis_obj_file,
            physicsClientId=self.client_id,
            rgbaColor=self.color,
        )  # type: ignore
        return super()._create_visual_geom(colMeshID, visMeshID, pb_collision)