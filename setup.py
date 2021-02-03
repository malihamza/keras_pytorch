from setuptools import setup

setup(
    name='pytorchkeras',
    version='0.0.1',
    description='Its a Keras replica for Pytorch',
    py_modules=["PytorchKeras", "Constants"],
    package_dir={'': 'src'},
)
