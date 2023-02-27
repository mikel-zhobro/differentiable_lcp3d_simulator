import os
import time

from PIL import Image

def create_gif(rgb_images, dir):
    imgs = [Image.fromarray(img) for img in rgb_images]
    imgs[0].save(
        dir + "visual.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0
    )


def get_experiment_dir(base_dir=None, ending=""):
    "Backups the data for reproduction, creates plots and saves plots."
    tmp = "/is/sg2/mzhobro/Desktop/Experiments/" if base_dir is None else base_dir
    directory = "{}{}/".format(tmp, time.strftime("%Y_%m_%d"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    dir_exp = directory + "{}-0".format(time.strftime("%H_%M_%S")) + ending

    i = 1
    while os.path.exists(dir_exp):
        dir_exp = dir_exp[:-1] + f"-{i}"
        i += 1

    dir_exp += "/"
    os.makedirs(dir_exp)

    return dir_exp

