import ode
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from .utils import Indices, get_tensor
from ..config import get_config

default_config = get_config()

X = Indices.X
Y = Indices.Y

x_axis = get_tensor([1.0, 0.0, 0.0])
y_axis = get_tensor([0.0, 1.0, 0.0])


class Contact:
    def __init__(
            self,
            normal: torch.Tensor,
            xa: torch.Tensor,
            xb: torch.Tensor,
            penetration: torch.Tensor,
            d1: torch.Tensor,
            d2: torch.Tensor,
            index1: int,
            index2: int,
            point: torch.Tensor,
            point_a_w: torch.Tensor,
            point_b_w: torch.Tensor,
            point_a_obj: torch.Tensor,
            point_b_obj: torch.Tensor,
    ) -> None:
        self.normal = normal  # normal in world frame, showing inside bodyA
        self.xa = xa  # vector from com of A to point of contact in world frame
        self.xb = xb
        self.penetration = penetration

        # tangential directions used to linearize friction and contact constraints
        self.d1 = d1
        self.d2 = d2

        # Index1 -> bodyA
        self.index1 = index1
        self.index2 = index2
        self.point = point  # contact point in world frame

        # Contact points in world frame
        self.point_a_w = point_a_w
        self.point_b_w = point_b_w

        # Contact points in object frame
        self.point_a_obj = point_a_obj
        self.point_b_obj = point_b_obj


class ContactHandler:
    def __init__(self):
        pass

    def update_contacts(self, p, R):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class OdeContactHandler(ContactHandler):
    def __init__(self, bodies):
        self.bodies = bodies
        # prepare ode
        # self.get_bodies = {}
        self.space = ode.HashSpace()
        for i, b in enumerate(bodies):
            b.geom_ode.body = i  # type: ignore
            # self.get_bodies[i] = b
            self.space.add(b.geom_ode)
        pass

    def update_contacts(self, p, R):
        contacts: list[Contact] = []
        self.space.collide([contacts, p, R], self)
        return contacts

    def __call__(self, args, geom1, geom2):
        contacts, p, R = args
        # print(geom1.body, geom2.body)
        if geom1 in geom2.no_contact:
            return

        # contacts: list[Contact] = args[0]
        ode_contacts = ode.collide(geom1, geom2)

        for c in ode_contacts:
            point, normal, penetration, geom1, geom2 = c.getContactGeomParams()
            penetration = get_tensor(penetration)

            normal = get_tensor(normal) # shows inside goem1
            point = get_tensor(point)

            penetration -= 2 * default_config.eps

            if penetration.item() < -2 * default_config.eps:
                return
            # assumes point is in the middle of the contacts on both bodies
            point_a_w = point + abs(penetration)/2 * normal
            point_b_w = point - abs(penetration)/2 * normal
            # Object poses
            R_a = get_tensor(geom1.getRotation()).view(3, 3)
            t_a = get_tensor(geom1.getPosition())
            R_b = get_tensor(geom2.getRotation()).view(3, 3)
            t_b = get_tensor(geom2.getPosition())
            # R_a = R[geom1.body]
            # t_a = p[geom1.body]
            # R_b = R[geom2.body]
            # t_b = p[geom2.body]

            # Contact points in object frame (detached)
            point_a_obj = R_a.T @ (point_a_w - t_a)
            point_b_obj = R_b.T @ (point_b_w - t_b)

            # Contact points and normal in world frame with relation to body pose
            # Use poses stored in Body class instead of the ones stored in ODE to make it differentiable
            xa = point_a_w - t_a
            xb = point_b_w - t_b

            val = torch.norm(torch.cross(normal, x_axis))
            if not val < 1e-9:
                d1 = torch.cross(torch.cross(normal, x_axis), normal)
                d1 = d1 / torch.norm(d1)
            else:
                d1 = torch.cross(torch.cross(normal, y_axis), normal)
            d2 = torch.cross(normal, d1)
            d2 = d2 / torch.norm(d2)

            contacts.append(
                Contact(
                    normal,         # [needed] normal in world frame, showing inside bodyA
                    xa,             # [needed] vector from com of A to point of contact A in world frame (incl penetration)
                    xb,             # [needed] vector from com of B to point of contact B in world frame (incl penetration)
                    penetration,    # [needed] penetration depth
                    d1,             # [needed] tangential directions used to linearize friction and contact constraints
                    d2,             # [needed] tangential directions used to linearize friction and contact constraints
                    geom1.body,     # [needed] Index1 -> bodyA
                    geom2.body,     # [needed] Index2 -> bodyB
                    point,          # [needed] contact point in world frame
                    point_a_w,      # [------] contact point A in world frame (incl penetration)
                    point_b_w,      # [------] contact point B in world frame (incl penetration)
                    point_a_obj,    # [------] contact point A in object frame (incl penetration)
                    point_b_obj,    # [------] contact point B in object frame (incl penetration)
                )
            )
        return contacts

    @staticmethod
    def unit_vector(f):
        return (
            f / torch.norm(f) if not torch.norm(f) < 1e-9 else f.new_tensor([0, 0, 0])
        )


def filter_contact_points(contacts: list[Contact]):
    num_contacts = len(contacts)
    if num_contacts <= 1:
        return

    # Check valid contacts
    contacts_valid_all = list(filter(lambda c: c.normal.norm() > 1e-12, contacts))
    contacts_valid_obj = split_object_pairs(contacts_valid_all)
    clusters_filtered_obj: list[list[Contact]] = []

    for contacts_valid in contacts_valid_obj:
        # Cluster contacts
        contact_clusters: list[list[Contact]] = []
        while len(contacts_valid) > 0:
            n = contacts_valid[0].normal
            max_val = n.new_tensor(1.0)
            cluster = list(
                filter(
                    lambda c: torch.acos(torch.min(torch.dot(c.normal, n), max_val))
                              < 1e-2,
                    contacts_valid,
                )
            )
            contact_clusters.append(cluster)
            contacts_valid = list(set(contacts_valid) - set(cluster))

        # Filter further points based by generating a (possibly lower-dimensional) convex hull
        clusters_filtered = list(map(process_cluster, contact_clusters))
        clusters_filtered_obj.append(sum(clusters_filtered, []))

    # Flatten the list and assign it to contacts
    contacts = sum(clusters_filtered_obj, [])


def split_object_pairs(contacts: list[Contact]):
    object_pairs: dict[tuple[int, int], list[Contact]] = {}
    for c in contacts:
        if not (
                (c.index1, c.index2) in object_pairs.keys()
                or (c.index2, c.index1) in object_pairs.keys()
        ):
            object_pairs[(c.index1, c.index2)] = []
        object_pairs[(c.index1, c.index2)].append(c)
    return list(object_pairs.values())


def process_cluster(cluster: list[Contact]):
    eps = default_config.eps
    completed = False
    points = torch.stack(list(map(lambda c: c.point, cluster))).detach()
    assert (
            points.ndimension() == 2
    ), f"Invalid number of dimensions. Expected 2, got {points.ndimension()}"
    while not completed:
        if points.shape[1] > 1:
            try:
                hull = ConvexHull(points.cpu().numpy())
                indices = points.new_tensor(hull.vertices).long()
                completed = True
            except QhullError:
                # Qhull didn't work, likely because the points only span a lower-dim space
                # => Remove dim with smallest variance
                variance = points.var(dim=0)
                mask = torch.ones(points.shape[1]).bool()
                mask[variance.argmin()] = False
                points = points[:, mask]
        else:
            # If we reduced the points to 1D convex hull is just min and max
            points_min, min_indices = points.min(dim=0)
            points_max, max_indices = points.max(dim=0)
            if points_max - points_min > eps:
                indices = torch.stack([min_indices, max_indices])
            else:
                indices = min_indices
            completed = True

    return [cluster[idx] for idx in indices]  # type: ignore
