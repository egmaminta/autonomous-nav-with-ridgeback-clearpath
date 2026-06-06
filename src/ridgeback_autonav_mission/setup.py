from setuptools import find_packages, setup

package_name = 'ridgeback_autonav_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mqi',
    maintainer_email='mnlmmnt@gmail.com',
    description='Deterministic mission FSM: parse task, explore, approach, confirm, return home.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mission_node = ridgeback_autonav_mission.mission_node:main',
        ],
    },
)
