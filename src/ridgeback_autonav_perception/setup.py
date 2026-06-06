from setuptools import find_packages, setup

package_name = 'ridgeback_autonav_perception'

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
    description='Room-sign perception: YOLO detector + PARSeq recognizer + depth-to-map projection.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'perception_node = ridgeback_autonav_perception.perception_node:main',
            'camera_publisher = ridgeback_autonav_perception.camera_publisher:main',
        ],
    },
)
