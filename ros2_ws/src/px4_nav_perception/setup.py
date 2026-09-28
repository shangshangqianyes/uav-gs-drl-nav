from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'px4_nav_perception'

setup(
    name=package_name,
    version='0.0.0',
    # Automatically include subpackages as well
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools', 'scipy', 'numpy<2'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='PX4 + ROS2 perception package (LiDAR)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grid_fuser = px4_nav_perception.nodes.grid_fuser:main',
            'odom_to_tf = px4_nav_perception.nodes.odom_to_tf:main',
        ],
    },
)

