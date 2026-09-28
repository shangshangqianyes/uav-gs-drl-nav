from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Args (only course is kept - used by goal_pub)
    course = LaunchConfiguration('course', default='R')

    goals_yaml = PathJoinSubstitution([
        FindPackageShare('px4_nav_planner'), 'config', 'goals.yaml'
    ])

    goal_pub = Node(
        package='px4_nav_planner',
        executable='course_goal_publisher_node',
        name='course_goal_publisher',
        output='screen',
        parameters=[
            goals_yaml,
            {'default_course': course},
            # Homing is performed by rl_offboard_node (keep this value identical to goals.yaml -- no-op).
            {'home_xyz': [5.0, 0.0, 4.0]},
            {'use_sim_time': True},
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='R'),
        goal_pub,
    ])

