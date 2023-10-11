# LCP Physics 3D
Differentiable 3D physics engine with ODE contact detection and PyBullet dynamics visualization based on [de avila Belbute-Peres et. al.](https://proceedings.neurips.cc/paper_files/paper/2018/hash/842424a1d0595b76ec4fa03c46e8d755-Abstract.html).

The repo should be self contained and can be used as a package.


## Organization
``` bash
Readme.md                               # this file
setup.py                                # setup file for the package


examples                                # examples
├── compare.py                          # functionalities to compare different runs for the sphere example
├── sphere_with_initialized_state.py    # sphere example with initialized state where we compare LCP and PyBullet
└── utils.py                            # utility functions for the examples

lcp3d                                   # main package
├── lcp                                 # lcp package
│   ├── solvers.py
│   ├── lcp.py
│   │   ├── batch.py
│   │   ├── cvxpy.py
│   │   ├── pdipm.py
│   │   └── spbatch.py
│   └── util.py
│
├── physics
│   ├── bodies.py                       # body class
│   ├── constraints.py                  # constraint class
│   ├── contacts.py                     # contact class
│   ├── engines.py                      # engine class
│   ├── forces.py                       # force class
│   ├── utils.py                        # utility functions
└── └── world.py                        # holds implementation of world and also most of the changes
```

## Installation
### a. Create conda environment

```bash
conda create -n my_env python=3.9
conda activate my_env
```

### b. Installing pytorch3d at
from within the virtual env we call
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
```

### c. Installation recipe for Open Dynamics Engine (ODE)

1. Download stable release from [https://bitbucket.org/odedevs/ode/downloads/](https://bitbucket.org/odedevs/ode/downloads/)
2. Unzip and install
3. (For MAC only) We get an error for type redifinition with different type -> Open the folder ode-0.15.3/..   in vscode and ctrl-shift-f `uint64` and replace all its occurences with `uint64_ode`
```bash
tar xf ode-0.16.2.tar.gz
cd ode-0.16.2
./configure --enable-double-precision --with-trimesh=opcode --enable-new-trimesh --enable-shared --prefix=$HOME/Installs/ode
make
make install
```

3. Add paths in .zshrc or .bashrc

```bash
export PKG_CONFIG_PATH="$HOME/Installs/ode/lib/pkgconfig"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/Installs/ode/lib/"
```

4. Enable python bindings

```bash
cd ode-0.16.2/bindings/python/
conda activate my_env  # environment for which you want to install the bindings
pip install .
```
