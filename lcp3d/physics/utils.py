import numpy as np
import torch

from typing import Union
import pytorch3d.transforms as pyt
from ..config import get_config


config = get_config()


class PyBulletConnection:
    def __init__(self):
        self.list_connections = []

    def new_connection(
        self,
        connection="gui",
        key=1234,
        networkAddress="localhost",
        networkPort=6667,
        options=None,
    ):
        import pybullet as pb
        if connection == "gui":
            clientIndex = pb.connect(pb.GUI)
            pb.resetDebugVisualizerCamera(
                cameraDistance=7.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=clientIndex,
            )
            self.list_connections.append(clientIndex)
            print("Bullet client created: GUI")
            return clientIndex
        elif connection == "direct":
            clientIndex = pb.connect(pb.DIRECT)
            self.list_connections.append(clientIndex)
            print("Bullet client created: DIRECT, ID: {}".format(clientIndex))
            return clientIndex
        elif connection == "shared_memory":
            clientIndex = pb.connect(pb.SHARED_MEMORY, key=key)
            self.list_connections.append(clientIndex)
            return clientIndex
        elif connection == "udp":
            clientIndex = pb.connect(
                pb.UDP, networkAddress=networkAddress, networkPort=networkPort  # type: ignore
            )
            self.list_connections.append(clientIndex)
            return clientIndex
        elif connection == "tcp":
            clientIndex = pb.connect(
                pb.TCP, networkAddress=networkAddress, networkPort=networkPort  # type: ignore
            )
            self.list_connections.append(clientIndex)
            return clientIndex
        else:
            raise ValueError("Invalid connection type")

    def disconnect(self, client_id=None):
        if client_id is None:
            for c in self.list_connections:
                pb.disconnect(c)
        else:
            for c in client_id:
                pb.disconnect(c)


class Indices:
    X = 0
    Y = 1
    Z = 2

    def __init__(self):
        pass


def rotation(orient, base_tensor):
    """Returns pose tensor of size [7,]
    Allows to provide only position, position and 3 elements of orientation or full pose."""
    orient = get_tensor(orient, base_tensor=base_tensor).squeeze()
    if orient.shape == (3, 3):
        rot = orient
    elif orient.shape[-1] == 3:
        rot = pyt.quaternion_to_matrix(quat(orient))
    elif orient.shape[-1] == 4:
        rot = pyt.quaternion_to_matrix(orient)
    else:
        raise Exception(f"Invalid position vector dimensions: {orient.shape[-1]}")
    return rot


def pose(pos, base_tensor):
    """Returns pose tensor of size [7,]
    Allows to provide only position, position and 3 elements of orientation or full pose."""
    pos = get_tensor(pos, base_tensor=base_tensor)
    if pos.shape[-1] == 3:
        p = torch.cat((pos, pos.new_tensor([1.0, 0.0, 0.0, 0.0])))
    elif pos.shape[-1] == 6:
        p = torch.cat((pos[:3], quat(pos[3:])))
        assert p.shape[0] == 7, p
    elif pos.shape[-1] == 7:
        p = pos
    else:
        raise Exception(f"Invalid position vector dimensions: {pos.shape[-1]}")
    return p.view(-1)


def velocity(vel, base_tensor):
    """Returns velocity tensor of size [6,]
    Allows to provide only position velocity or full velocity."""
    vel = get_tensor(vel, base_tensor=base_tensor)
    if vel.shape[0] == 3:
        v = torch.cat([vel, vel.new_zeros((3,) + vel.shape[1:])])
    elif vel.shape[0] == 6:
        v = vel
    else:
        raise Exception("Invalid velocity vector dimensions")
    return v


def get_directions(d1, d2, depth=2):
    if not depth > 0:
        raise ValueError("Depth must always be greater than zero.")
    if depth == 1:
        return [d1, -d1]
    elif depth == 2:
        return [d1, -d1, d2, -d2]
    else:  # depth == 3:
        # TODO: Write a better algorithm
        d3 = d1 + d2
        d3 = d3 / torch.norm(d3)
        d4 = d1 - d2
        d4 = d4 / torch.norm(d4)
        return [d1, -d1, d2, -d2, d3, -d3, d4, -d4]


def J(c, *dirs, num_dirs=None):
    if num_dirs is not None:
        if not num_dirs == len([*dirs]):
            raise ValueError(
                f"Number of input directions ({len([*dirs])}) do not match number of fric_dirs ({num_dirs})"
            )
    else:
        num_dirs = len([*dirs])

    if not num_dirs % 2 == 0:
        raise ValueError("Number of friction directions should be even")
    return torch.stack(
        # [torch.cat([torch.cross(c, direction), direction]) for direction in [*dirs]]
        [torch.cat([direction, torch.cross(c, direction)]) for direction in [*dirs]]
    )

def get_quat_xyzw_numpy(q: np.ndarray):
    assert q.shape[-1] == 4
    return np.concatenate((q[1:], q[:1]), axis=-1)

def get_quat_xyzw(q: torch.Tensor):
    assert q.shape[-1] == 4
    return torch.cat((q[..., 1:], q[..., :1]), dim=-1)


def get_quat_wxyz(q: torch.Tensor):
    assert q.shape[-1] == 4
    return torch.cat((q[..., -1:], q[..., :-1]), dim=-1)


def quat(ang, style="wxyz"):
    """Returns a quaternion, and takes euler angles as inputs"""
    if isinstance(ang, list):
        assert len(ang) == 3
        ang = get_tensor(ang)
    elif isinstance(ang, torch.Tensor):
        assert ang.shape[-1] == 3
    else:
        raise TypeError(
            "Unknown datatype for angles. Expected either list or torch.Tensor, got {}".format(
                type(ang)
            )
        )

    rot_mat = pyt.euler_angles_to_matrix(ang, "XYZ")
    quaternion = pyt.matrix_to_quaternion(
        rot_mat
    )  # returns quaternion with real-part first
    if style == "xyzw":
        quaternion = get_quat_xyzw(quaternion)
    return quaternion


def check_quat(q: torch.Tensor) -> bool:
    return q.shape[-1] == 4


def get_tensor(x, base_tensor: Union[torch.Tensor, None] = None, **kwargs):
    """Wrap array or scalar in torch Tensor, if not already."""
    if isinstance(x, torch.Tensor):
        return x
    elif base_tensor is not None:
        return base_tensor.new_tensor(x, **kwargs)
    else:
        return torch.tensor(
            x,
            dtype=config.dtype,
            device=config.lcp_device,
            # layout=Params.DEFAULT_LAYOUT,
            **kwargs,
        )


def rev_comp(t):
    assert t.shape[-1] == 6


def se3_add_tau(p, tau):
    pass


def process_vhacd_meshes(obj_file: str):
    with open(obj_file, "r") as f:
        obj_msh_lines = f.readlines()
    # Calculate number of convex objects
    obj_lines = []

    # First entry
    start_line, num_line = 0, 0  # make mypy happy
    for num_line, line in enumerate(obj_msh_lines):
        entries = line.rstrip().split(" ")
        if entries[0] == "o":
            start_line = num_line
            break

    # Iterate through all the lines
    for num_line, line in enumerate(obj_msh_lines):
        entries = line.rstrip().split(" ")
        if entries[0] == "o":
            end_line = num_line
            if not start_line == end_line:
                obj_lines.append((start_line, end_line))
            start_line = num_line

    end_line = num_line
    if not start_line == end_line:
        obj_lines.append((start_line, end_line))

    vertices = []
    faces = []

    for num, (start_line, end_line) in enumerate(obj_lines):
        # print(obj_msh_lines[start_line:end_line])
        _vertices, _faces = get_vertices_faces(obj_msh_lines[start_line:end_line])
        vertices += _vertices
        faces += _faces

    return vertices, faces


def get_mesh_params(obj_file: str):
    with open(obj_file, "r") as f:
        obj_msh_lines = f.readlines()
    vertices, faces = get_vertices_faces(obj_msh_lines)
    return vertices, faces


def get_vertices_faces(obj_msh_lines: list):
    vertices = []
    faces = []
    for line in obj_msh_lines:
        entries = line.rstrip().split(" ")
        # Check vertices
        if entries[0] == "v":
            # Check x-coordinates
            assert len(entries[1:]) == 3
            vertices.append([float(e) for e in entries[1:]])
        elif entries[0] == "f":
            # Check x-coordinates
            assert len(entries[1:]) == 3
            faces.append([int(e.split("/")[0]) - 1 for e in entries[1:]])
    return vertices, faces


def update_lims(lims: list, val):
    if val < lims[0]:
        lims[0] = val

    if val > lims[1]:
        lims[1] = val


def mesh_bbox(obj_file: str, x_c: bool = True, y_c: bool = True, z_c: bool = True):
    with open(obj_file, "r") as f:
        obj_msh_lines = f.readlines()

    x_lims = [0.0, 0.0]
    y_lims = [0.0, 0.0]
    z_lims = [0.0, 0.0]

    for line in obj_msh_lines:
        entries = line.rstrip().split(" ")
        # Check vertices
        if entries[0] == "v":
            # Check x-coordinates
            assert len(entries[1:]) == 3
            x, y, z = [float(e) for e in entries[1:]]
            if x_c:
                update_lims(x_lims, x)
            if y_c:
                update_lims(y_lims, y)
            if z_c:
                update_lims(z_lims, z)

    return x_lims, y_lims, z_lims


def mesh_center(obj_file: str):
    x_lims, y_lims, z_lims = mesh_bbox(obj_file)
    return [
        0.5 * (x_lims[0] + x_lims[1]),
        0.5 * (y_lims[0] + y_lims[1]),
        0.5 * (z_lims[0] + z_lims[1]),
    ]


def get_instance(mod, class_id):
    """Checks if class_id is a string and if so loads class from module;
    else, just instantiates the class."""
    if isinstance(class_id, str):
        # Get by name if string
        return getattr(mod, class_id)()
    else:
        # Else just instantiate
        return class_id()


def dist_from_bottom(obj_file: str):
    x_lims, y_lims, z_lims = mesh_bbox(obj_file, x_c=False, y_c=False, z_c=True)
    return -z_lims[0] + 0.5 * (z_lims[0] + z_lims[1])


def check_rot(rot: torch.Tensor):
    print(
        f"R*R'={torch.allclose(rot @ rot.T, torch.eye(3, dtype=rot.dtype, device=rot.device))}, ||R|| = {rot.norm(dim=0)}"
    )
    pass


def p_hom_transformation(t: torch.Tensor, q: torch.Tensor):
    return torch.cat(
        (
            torch.cat((pyt.quaternion_to_matrix(q), t[..., None]), dim=-1),
            t.new_tensor([0, 0, 0, 1])[None, ...],
        ),
        dim=0,
    )
