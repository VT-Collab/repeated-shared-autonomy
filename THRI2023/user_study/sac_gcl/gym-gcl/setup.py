from setuptools import setup

setup(name="gym_gcl",
      version="0.1",
      author="Collab",
      packages=["gym_gcl", "gym_gcl.envs"],
      install_requires = ["gym", "numpy"]
)
