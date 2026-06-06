from setuptools import find_packages, setup

package_name = 'ridgeback_autonav_nav'

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
    description='From-scratch mapping, planning, holonomic control, and frontier exploration.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mapping_node = ridgeback_autonav_nav.mapping_node:main',
            'nav_server_node = ridgeback_autonav_nav.nav_server_node:main',
            'explorer_node = ridgeback_autonav_nav.explorer_node:main',
        ],
    },
)
