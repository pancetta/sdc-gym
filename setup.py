from setuptools import setup

setup(
    name='sdc_gym',
    version='0.0.2',
    install_requires=[
        'gym',
        'pySDC',
        'numpy',
        'scipy',
        'matplotlib',
        'stable_baselines',
        'pyzt',
        'jax',
        'jaxlib',
        'optax',
    ],
)
