from typing import Union


class Config:
    def __init__(self, config_name: Union[str, None] = None):
        if config_name is not None:
            self.config_name = config_name
        else:
            self.config_name = "config"

        # Set the default config
        self.g = -10
        # Contact detection parameter
        self.contact_handler = "OdeContactHandler"
        self.filter_contacts = True
        self.correct_initial_penetration = True
        self.strict_no_penetration = True
        self.num_sub_steps = 1
        self.eps = 1e-3
        self.tol = 1e-8  # Penetration tolerance parameter

        # Physics parameters
        self.dim = 3
        self.restitution = 0.2
        self.fric_coeff = 0.2
        self.fric_dirs = 4

        # Default simulation parameters
        self.engine = "QPEngine"
        self.fps = 50
        self.dt = 1.0 / self.fps
        self.post_stabilization = False
        # self.solver = "PDIPM_BATCHED"
        self.solver = "PDIPM_BATCHED"

        import torch
        self.dtype = torch.double
        # self.device = torch.device("cuda")
        # self.lcp_device = torch.device("cuda")
        self.device = torch.device("cpu")
        self.lcp_device = torch.device("cpu")
        self.layout = torch.strided

        # compute = "local"
        compute = "cluster"

        if compute == "local":
            # Datasets local
            self.ycb_video_dataset_dir = "/is/rg/ev/scratch/datasets/YCB_Video_Dataset/"
            self.vhacd_meshes_dir = (
                "/is/sg2/rkandukuri/my_code/ekf_physics/ekf_physics/vhacd_meshes/"
            )
            self.syn_phys_dataset_dir = "/is/rg/ev/scratch/datasets/synphys/"

            # FFB6D checkpoint local
            self.checkpoint = "/is/rg/ev/scratch/shared/CVPR2022_physicstrack/ffb6d/train_log/synphys/checkpoints/FFB6D_best.pth.tar"
            # self.checkpoint = "/is/rg/ev/scratch/shared/CVPR2022_physicstrack/ffb6d/train_log/synphys_inv_11746700/checkpoints/FFB6D_best.pth.tar"

        elif compute == "cluster":
            # Datasets cluster
            self.ycb_video_dataset_dir = (
                "/is/cluster/work/projects/is-rg-ev/datasets/YCB_Video_Dataset/"
            )
            self.vhacd_meshes_dir = "/is/cluster/work/rkandukuri/datasets/vhacd_meshes/"
            self.syn_phys_dataset_dir = (
                "/is/cluster/work/projects/is-rg-ev/datasets/synphys/"
            )

            # FFB6D checkpoint cluster
            self.checkpoint = "/is/cluster/work/rkandukuri/inference/ffb6d/synphys/november_model/FFB6D_best.pth.tar"

        else:
            raise NotImplementedError(f"Compute not implemented for {compute}")

    def add_params(self, **kwargs):
        for key, val in kwargs:
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown config {val}")

    @property
    def solver_id(self):
        if self.solver == "PDIPM_BATCHED":
            return 1
        elif self.solver == "CVXPY":
            return 2
        else:
            raise ValueError(f"Unknown solver {self.solver}!")
