from setuptools import setup, find_packages

print(find_packages())

# write a setup.py file for myt lcp3d package
setup(
    name="lcp3d",
    version="0.1",
    description="LCP3D",
    author="Mikel",
    packages=find_packages(),
    install_requires=["numpy", "cython", "cvxpy", "scipy", "matplotlib","pybullet"],
)