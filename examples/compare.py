import matplotlib.pyplot as plt
import pickle


def plot(dir=None, lcp_tuple=None, bullet_tuple=None):
    if dir is None:
        if lcp_tuple is None or bullet_tuple is None:
            print("Please provide a directory or a tuple of data.")
            return
        poses2, vel2, times2, Ias2 = lcp_tuple
        poses3, vel3, times3, Ias3 = bullet_tuple
    else:
        with open(dir + "se3.dat", "rb") as file:
            poses2 = pickle.load(file)
            vel2 = pickle.load(file)
            times2 = pickle.load(file)
            Ias2 = pickle.load(file)
        with open(dir + "bullet.dat", "rb") as file:
            poses3 = pickle.load(file)
            vel3 = pickle.load(file)
            times3 = pickle.load(file)
            Ias3 = pickle.load(file)


    _plot(dir, poses2, poses3, times2, times3)
    # _plot_vel(dir, vel1-vel3, vel2-vel3, vel3-vel3)
    _plot_vel(dir, vel2, vel3)
    # plot_energy(dir, vel2, vel3, Ias2, Ias3)
    plt.show()


def _plot(dir, poses2, poses3, times2, times3, block=False):
    fig = plt.figure(figsize=(16, 16), dpi=80)
    ax = fig.add_subplot(421)
    ax.plot(poses2[:, 0], "--")
    ax.plot(poses3[:, 0])
    # plt.plot(poses2[:,0] - poses1[:,0])
    ax.set_title("x", fontsize=14)

    ax = fig.add_subplot(423)
    ax.plot(poses2[:, 1], "--")
    ax.plot(poses3[:, 1])
    # ax.plot(poses2[:,1] - poses1[:,1])
    ax.set_title("y", fontsize=14)

    ax = fig.add_subplot(425)
    ax.plot(poses2[:, 2], "--")
    ax.plot(poses3[:, 2])
    # ax.plot(poses2[:,2] - poses1[:,2])
    ax.set_title("z", fontsize=14)

    # Orientation
    ax = fig.add_subplot(422)
    ax.plot(poses2[:, 3], "--")
    ax.plot(poses3[:, 6])
    ax.set_title("q_w", fontsize=14)

    ax = fig.add_subplot(424)
    ax.plot(poses2[:, 4], "--")
    ax.plot(poses3[:, 3])
    ax.set_title("q_x", fontsize=14)

    ax = fig.add_subplot(426)
    ax.plot(poses2[:, 5], "--")
    ax.plot(poses3[:, 4])
    ax.set_title("q_y", fontsize=14)

    ax1 = fig.add_subplot(428)
    ax1.plot(poses2[:, 6], "--")
    ax1.plot(poses3[:, 5])
    ax1.set_title("q_z", fontsize=14)

    ax = fig.add_subplot(427)
    l2 = ax.plot(times2, "--")[0]
    l3 = ax.plot(times3)[0]

    ax.axhline(
        y=sum(times2) / len(times2), color=l2.get_color(), linestyle=l2.get_linestyle()
    )
    ax.axhline(
        y=sum(times3) / len(times3), color=l3.get_color(), linestyle=l3.get_linestyle()
    )
    ax.set_title("Times", fontsize=14)

    labels = ["LCP", "BULLET"]
    plt.figlegend(
        [l2, l3], labels, loc="upper center", ncol=5, labelspacing=0.0, fontsize=20
    )
    # plt.suptitle("LCP_SO3, LCP_SE3, BULLET comparison", fontsize=20)

    plt.suptitle("Positions and orientations", y=0.93, fontsize=24)
    if dir is not None:
        plt.savefig(dir + "plot_poses.png", bbox_inches="tight")
    plt.show(block=block)


def _plot_vel(dir, vel2, vel3, block=False):
    fig = plt.figure(figsize=(16, 16), dpi=80)
    ax = fig.add_subplot(321)
    ax.plot(vel2[:, 0], "--")
    ax.plot(vel3[:, 0])
    # plt.plot(poses2[:,0] - poses1[:,0])
    ax.set_title("vx", fontsize=14)

    ax = fig.add_subplot(323)
    ax.plot(vel2[:, 1], "--")
    ax.plot(vel3[:, 1])
    # ax.plot(poses2[:,1] - poses1[:,1])
    ax.set_title("vy", fontsize=14)

    ax = fig.add_subplot(325)
    ax.plot(vel2[:, 2], "--")
    ax.plot(vel3[:, 2])
    # ax.plot(poses2[:,2] - poses1[:,2])
    ax.set_title("vz", fontsize=14)

    # Orientation
    ax = fig.add_subplot(322)
    ax.plot(vel2[:, 3], "--")
    ax.plot(vel3[:, 3])
    ax.set_title("wx", fontsize=14)

    ax = fig.add_subplot(324)
    ax.plot(vel2[:, 4], "--")
    ax.plot(vel3[:, 4])
    ax.set_title("wy", fontsize=14)

    ax = fig.add_subplot(326)
    l2 = ax.plot(vel2[:, 5], "--")[0]
    l3 = ax.plot(vel3[:, 5])[0]
    ax.set_title("wz", fontsize=14)

    labels = ["LCP", "BULLET"]
    plt.figlegend(
        [l2, l3], labels, loc="upper center", ncol=5, labelspacing=0.0, fontsize=20
    )
    plt.suptitle("Velocities", y=0.93, fontsize=24)
    # plt.suptitle("LCP_SO3, LCP_SE3, BULLET comparison", fontsize=20)

    if dir is not None:
        plt.savefig(dir + "plot_velocities.png", bbox_inches="tight")
    plt.show(block=block)


def plot_energy(dir, vels2, vels3, Ias2, Ias3):
    plt.figure()
    l2 = plt.plot([vel.T @ Ia @ vel for Ia, vel in zip(Ias2, vels2[:, 3:])], "--")[0]
    l3 = plt.plot([vel.T @ Ia @ vel for Ia, vel in zip(Ias3, vels3[:, 3:])])[0]
    labels = ["LCP", "BULLET"]
    plt.figlegend(
        [l2, l3], labels, loc="upper center", ncol=5, labelspacing=0.0, fontsize=20
    )
    plt.title("Energy")
    # plt.plot(np.linalg.norm([ Ia@vel for Ia, vel in zip(Ias1, vels1)], axis=-1))
    # plt.ylim(ymin=0, ymax=3)
    if dir is not None:
        plt.savefig(dir + "plot_energy.png", bbox_inches="tight")
    plt.show()


# if __name__ == "__main__":
#     dir = "/is/sg2/mzhobro/Desktop/Experiments/2022_06_20/17_04_27/"
#     # dir = "/is/sg2/mzhobro/Desktop/Experiments/2022_06_20/17_05_50/"
#     dir = "/is/sg2/mzhobro/Desktop/Experiments/2022_06_20/17_51_06/"
#     plot(dir)
