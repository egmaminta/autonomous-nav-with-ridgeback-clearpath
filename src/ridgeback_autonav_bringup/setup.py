import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ridgeback_autonav_bringup'

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
    description='Launch files, parameters, and the cross-machine topic relay for the ridgeback_autonav stack.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'topic_relay = ridgeback_autonav_bringup.topic_relay:main',
        ],
    },
)
