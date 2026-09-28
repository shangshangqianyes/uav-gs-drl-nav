from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_path

def generate_launch_description():
    # Package share path
    pkg_share = FindPackageShare('px4_nav_perception')
    # Actual path in the install space (for reading files)
    pkg_dir = get_package_share_path('px4_nav_perception')

    # Pass the parameter/rviz config files as Substitution paths
    params   = PathJoinSubstitution([pkg_share, 'config', 'grid_fuser.yaml'])
    rviz_cfg = PathJoinSubstitution([pkg_share, 'rviz', 's3_local_mapping.rviz'])

    # Read the actual URDF file and inject it as a string
    urdf_file = pkg_dir / 'urdf' / 'iris_rplidar_depth.urdf'
    robot_description_str = urdf_file.read_text()

    use_sim_time = {'use_sim_time': True}
    # rviz:=false -> headless (for batch experiments). Default true (keeps GUI for manual verification)
    rviz_on = LaunchConfiguration('rviz')

    # 1) Broadcast the URDF as TF (base_link -> rplidar_link, camera_link, etc.)
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_str,  # inject the string directly
            'use_sim_time': True
        }],
    )

    # 2) MAVROS Odom -> TF (odom->base_link) bridge
    odom_to_tf = Node(
        package='px4_nav_perception',
        executable='odom_to_tf',
        name='odom_to_tf',
        output='screen',
        parameters=[
            use_sim_time,
            {
                'odom_topic': '/mavros/local_position/odom',
                'pose_topic': '/mavros/local_position/pose',
                'parent_frame': 'odom',
                'child_frame': 'base_link',
            }
        ],
        respawn=True, respawn_delay=1.0
    )

    # 3) OccupancyGrid generation node
    grid_fuser = Node(
        package='px4_nav_perception',
        executable='grid_fuser',
        name='grid_fuser',
        output='screen',
        parameters=[params, use_sim_time],
        respawn=True, respawn_delay=1.0
    )

    # 4) RViz2 (headless batch execution possible with rviz:=false)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],   # remove this argument if there is no rviz config file
        output='screen',
        parameters=[use_sim_time],
        condition=IfCondition(rviz_on),
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                             description='true: run RViz2 GUI, false: headless'),
        robot_state_pub,
        odom_to_tf,
        grid_fuser,
        rviz,
    ])
