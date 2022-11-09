from setuptools import setup, find_packages

setup(
    name='hundredandten',
    version='0.1.0',
    description='HundredAndTen Gym Environment',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.4,<=0.15.7',
        'numpy>=1.13.0',
        'opencv-python>=3.4.2.0',
        'hundredandten>=0.3.0'
    ]
)


