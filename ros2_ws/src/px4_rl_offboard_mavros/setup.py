from setuptools import setup
import os
from glob import glob

package_name = 'px4_rl_offboard_mavros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/rl_offboard.launch.py']),
        ('share/' + package_name + '/config', [
            'config/params.yaml',
            'config/params_astar.yaml',
            'config/params_ppo.yaml',
            'config/params_fm2.yaml',
            'config/params_sac.yaml',
        ]),
        ('share/' + package_name + '/models', glob('models/*')),
    ],
    install_requires=['setuptools', 'scipy'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='RL Offboard Control via MAVROS',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'rl_offboard_node = px4_rl_offboard_mavros.rl_offboard_node:main'
        ],
    },
)
