import pickle
import time

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

import pybullet as pb

from lcp3d import World, get_config
from lcp3d.physics.constraints import TotalConstraint
from lcp3d.physics.forces import MultiWrench, F
from lcp3d.physics.bodies import RigidBody, Box, Sphere
from lcp3d.physics.utils import get_tensor, PyBulletConnection

from utils import get_experiment_dir, create_gif

from examples.wrappers.lcp_3d_wrapper import (
    LCP_3D_DynamicModel as WorldWrapper,
)

Defaults = get_config()

def make_world(client_id, dt: float = Defaults.dt, pb_coll=False):
    bodies: list[RigidBody] = []
    joints = []
    # Plane
    plane = Box(
        [0, 0, -0.25], 50, 50, 0.5, mass=500, fric_coeff=0.0, client_id=client_id, pb_collision=pb_coll,
    )
    bodies.append(plane)
    joints.append(TotalConstraint(plane))

    # Box1
    box1 = Box(
        [0.0, 0, 2.25],
        # [0.0, 0, 0.25],
        length=0.5,
        width=1,
        height=0.5,
        mass=4,
        # vel=(1, 1, 1, 1, 1, 1),
        fric_coeff=0.2,
        client_id=client_id,
        color=(0.0, 0.8, 0.0, 1),
        pb_collision=pb_coll,
    )
    bodies.append(box1)

   # Box1
    box = Sphere(
        [0.0, 0.1, 0.25],
        radius=0.5,
        mass=4,
        # vel=(1, 1, 1, 1, 1, 1),
        fric_coeff=0.2,
        client_id=client_id,
        color=(0.0, 0.8, 0.0, 1),
        pb_collision=pb_coll,
    )
    bodies.append(box)

    multiforces = [MultiWrench(1, [F(F.TWrench)])]
    ixs = [1]
    world = World(bodies, constraints=joints, dt=dt)  # type: ignore

    return world, bodies, client_id, multiforces, ixs


class LCPExperiment:
    def __init__(self, time, fps=50, mode="gui"):
        self.fps = fps
        self.dt = 1.0 / self.fps
        self.total_time_steps = int(time / self.dt)
        self.client_id = PyBulletConnection().new_connection(mode)
        self.multiforces: list[MultiWrench]
        self.ixs: list[int]

    def make_world(self, gt_forces, lcp_engine=True):
        self.world, self.bodies, self.clientID, self.multiforces, self.ixs = make_world(self.client_id, self.dt, pb_coll=not lcp_engine)
        if lcp_engine:
            def step_func(force):
                self.world.add_force(force, self.multiforces[0])
                self.world.step(True)
            get_pose = lambda b: b.get_pose()
            get_vel = lambda b: b.get_vel()
            get_IA = lambda b: b.ang_inert_world
        else:
            pb.setTimeStep(self.dt, physicsClientId=self.clientID)
            pb.createConstraint(
                self.bodies[0].get_geom_bullet(),
                -1,
                -1,
                -1,
                pb.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                self.bodies[0].position.cpu().detach().numpy(),
                # [0, 0, -0.025],
                physicsClientId=self.clientID,
            )  # type: ignore
            pb.setGravity(0, 0, -10, physicsClientId=self.clientID)

            def step_func(force):
                ix_bull = self.bodies[self.ixs[0]].get_geom_bullet()
                pb.applyExternalForce (ix_bull, -1, posObj=[0,0,0], forceObj=force[:3].tolist(), flags=pb.LINK_FRAME, physicsClientId=self.clientID)
                pb.applyExternalTorque(ix_bull, -1, torqueObj=force[3:].tolist(), flags=pb.LINK_FRAME, physicsClientId=self.clientID)
                pb.stepSimulation(physicsClientId=self.clientID)
                time.sleep(0.001)

            def get_pose(b):
                p, o = pb.getBasePositionAndOrientation(
                    b.get_geom_bullet(), physicsClientId=self.clientID
                )
                return np.concatenate([p, o])

            def get_vel(b):
                v, w = pb.getBaseVelocity(
                    b.get_geom_bullet(), physicsClientId=self.clientID
                )
                return np.concatenate([v, w])

            def get_IA(b):
                R = pb.getMatrixFromQuaternion(
                    pb.getBasePositionAndOrientation(
                        b.get_geom_bullet(), physicsClientId=self.clientID
                    )[1]  # type: ignore
                )
                R = np.stack(R).reshape(3, 3)
                return (
                    R
                    @ np.stack(pb.calculateMassMatrix(b.get_geom_bullet(), []))[:3, :3]
                    @ R.T
                )

        return step_func, get_pose, get_vel, get_IA

    def run_experiment(self, gt_forces, lcp_engine=True):
        step, get_pose, get_vel, get_IA = self.make_world(gt_forces, lcp_engine)
        Ias = []
        rgb_images = []
        traj = []
        traj_velo = []
        times = []

        traj.append(get_pose(self.bodies[-1]))
        traj_velo.append(get_vel(self.bodies[-1]))
        Ias.append(get_IA(self.bodies[-1]))
        for t in tqdm(range(self.total_time_steps)):
            t_start = time.time()
            # self.world.add_force(gt_forces[0], self.multiforces[0])
            step(gt_forces[0])
            times.append(time.time() - t_start)

            traj.append(get_pose(self.bodies[-1]))
            traj_velo.append(get_vel(self.bodies[-1]))
            Ias.append(get_IA(self.bodies[-1]))
            # traj.append(np.cat([get_pose(b) for b in self.bodies[2:]]))
            # traj_velo.append(np.cat([get_vel(b) for b in self.bodies[2:]]))
            # rgb_images.append(pb.getCameraImage(200, 320, renderer=pb.ER_BULLET_HARDWARE_OPENGL)[2])

        print(f"TIME FOR WHOLE SIMULATION IS: {sum(times)}")
        return (
            np.stack(traj),
            np.stack(traj_velo),
            times,
            rgb_images,
            np.stack(Ias),
        )


if __name__ == "__main__":
    SAVE = False
    BULLET = True
    t = 1
    lcp = LCPExperiment(time=t, fps=240, mode="gui")
    f = torch.tensor([3.0, 0, 0, 0, 0, 0], dtype=Defaults.dtype, device=Defaults.device)[
        None, :
    ]
    f_t = f.repeat(int(t * 60), 1)


    print("------ SE3 ------")
    poses_se3, vel_se3, times_se3, rgb_images_se3, Ias_se3 = lcp.run_experiment(f_t)

    dir = None
    if SAVE:
        dir = get_experiment_dir()
        with open(dir + "se3.dat", "wb") as filehandler:
            pickle.dump(poses_se3, filehandler)
            pickle.dump(vel_se3, filehandler)
            pickle.dump(times_se3, filehandler)
            pickle.dump(Ias_se3, filehandler)

    if BULLET:
        print("------ BULLET ------")
        (
            poses_bullet,
            vel_bullet,
            times_bullet,
            rgb_images_bullet,
            Ias_bullet,
        ) = lcp.run_experiment(f_t, lcp_engine=False)
        if SAVE:
            with open(dir + "bullet.dat", "wb") as filehandler:
                pickle.dump(poses_bullet, filehandler)
                pickle.dump(vel_bullet, filehandler)
                pickle.dump(times_bullet, filehandler)
                pickle.dump(Ias_bullet, filehandler)



        # frames = rgb_images_bullet + rgb_images_so3 + rgb_images_se3
        # create_gif(frames, dir+"visualization")

        # # Plot
        from compare import plot
        plot(dir, (poses_se3, vel_se3, times_se3, Ias_se3), (poses_bullet, vel_bullet, times_bullet, Ias_bullet))