# LCP Physics 3D

Differentiable 3D physics engine with ODE contact detection, PyBullet dynamics visualization, and Tensorboard performance visualizatio



## Installation recipe for Open Dynamics Engine (ODE)

1. Download stable release from [https://bitbucket.org/odedevs/ode/downloads/](https://bitbucket.org/odedevs/ode/downloads/)
2. Unzip and install

```bash
tar xf ode-0.16.2.tar.gz
cd ode-0.16.2
./configure --enable-double-precision --with-trimesh=opcode --enable-new-trimesh --enable-shared --prefix=$HOME/Installs/ode
make
make install
```

3. Add paths in .zshrc or .bashrc

```bash
export PKG_CONFIG_PATH="/is/sg2/mzhobro/Installs/ode/lib/pkgconfig"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/is/sg2/mzhobro/Installs/ode/lib/"
```

4. Enable python bindings

```bash
cd ode-0.16.2/bindings/python/
conda activate my_env  # environment for which you want to install the bindings
pip install .
```
