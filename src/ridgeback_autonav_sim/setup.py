import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ridgeback_autonav_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mqi',
    maintainer_email='mnlmmnt@gmail.com',
    description='Self-contained 2D Ridgeback simulator for testing the ridgeback_autonav stack without hardware.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'sim_node = ridgeback_autonav_sim.sim_node:main',
            'fake_perception_node = ridgeback_autonav_sim.fake_perception_node:main',
        ],
    },
)
