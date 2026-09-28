#!/usr/bin/env python3
"""Launch file for running the RL Offboard node.
course_goal_publisher publishes R/L course goals to /planner/goal.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    course = LaunchConfiguration('course', default='R')

    goals_yaml = PathJoinSubstitution([
        FindPackageShare('px4_nav_planner'), 'config', 'goals.yaml'
    ])

    params_file = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params.yaml'
    ])
    params_astar_overlay = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params_astar.yaml'
    ])
    params_ppo_overlay = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params_ppo.yaml'
    ])
    params_fm2_overlay = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params_fm2.yaml'
    ])
    params_sac_overlay = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'config', 'params_sac.yaml'
    ])
    models_dir = PathJoinSubstitution([
        FindPackageShare('px4_rl_offboard_mavros'), 'models'
    ])
    planner_only = LaunchConfiguration('planner_only', default='false')
    rl_only = LaunchConfiguration('rl_only', default='false')
    # overlay: '' (default hybrid) | 'fm2' (the proposed method) | 'sac' (SAC-only baseline)
    # mutually exclusive with planner_only/rl_only (legacy astar/ppo) -- priority: overlay > rl_only > planner_only
    overlay = LaunchConfiguration('overlay', default='')

    # Option A (single flow): course_goal_publisher is a pure configuration node -- publishes only /planner/goal.
    # homing + OFFBOARD + ARM + velocity control are all performed by rl_offboard_node as a single velocity setpoint
    # stream (eliminates conflicts with the position plugin and OFFBOARD drops).
    goal_pub = Node(
        package='px4_nav_planner',
        executable='course_goal_publisher_node',
        name='course_goal_publisher',
        output='screen',
        parameters=[
            goals_yaml,
            {'default_course': course},
            {'enable_homing': True},
            {'use_sim_time': True},
        ],
    )

    rl_node = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}],
    )
    rl_node_planner_only = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_astar_overlay, {'use_sim_time': True}],
    )
    rl_node_rl_only = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_ppo_overlay, {'use_sim_time': True}],
    )
    # FM2 (the proposed method): params.yaml + params_fm2.yaml (nav_mode/replan/alpha)
    rl_node_fm2 = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_fm2_overlay, {'use_sim_time': True}],
    )
    # SAC-only: params.yaml + params_sac.yaml(nav_mode=ppo) + SAC model paths (params dict takes precedence after yaml)
    rl_node_sac = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_sac_overlay,
                    {'use_sim_time': True,
                     'policy_path': PathJoinSubstitution([models_dir, 'policy_sac_ts.pt']),
                     'obs_norm_path': PathJoinSubstitution([models_dir, 'obs_norm_sac.json'])}],
    )
    # C2_dt05 PPO-only (10M, separate checkpoint): params_ppo(nav_mode=ppo) + C2_dt05 model paths
    rl_node_c2dt05 = Node(
        package='px4_rl_offboard_mavros',
        executable='rl_offboard_node',
        name='rl_offboard_node',
        output='screen',
        parameters=[params_file, params_ppo_overlay,
                    {'use_sim_time': True,
                     'policy_path': PathJoinSubstitution([models_dir, 'policy_ppo_c2dt05_ts.pt']),
                     'obs_norm_path': PathJoinSubstitution([models_dir, 'obs_norm_ppo_c2dt05.json'])}],
    )
    planner_only_cond = PythonExpression(["'", planner_only, "' == 'true'"])
    rl_only_cond = PythonExpression(["'", planner_only, "' != 'true' and '", rl_only, "' == 'true' and '", overlay, "' == ''"])
    default_cond = PythonExpression(["'", planner_only, "' != 'true' and '", rl_only, "' != 'true' and '", overlay, "' == ''"])
    fm2_cond = PythonExpression(["'", overlay, "' == 'fm2'"])
    sac_cond = PythonExpression(["'", overlay, "' == 'sac'"])
    c2dt05_cond = PythonExpression(["'", overlay, "' == 'c2dt05'"])

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='R',
                             description='R(ight) or L(eft) course'),
        DeclareLaunchArgument('planner_only', default_value='false',
                             description='true: global planner (A*) only, RL disabled'),
        DeclareLaunchArgument('rl_only', default_value='false',
                             description='true: RL only, global planner disabled'),
        DeclareLaunchArgument('overlay', default_value='',
                             description="''(hybrid) | 'fm2' (FM2+PPO, the proposed method) | 'sac' (SAC-only baseline) | 'c2dt05' (C2_dt05 PPO-only 10M)"),
        goal_pub,
        GroupAction([rl_node], condition=IfCondition(default_cond)),
        GroupAction([rl_node_planner_only], condition=IfCondition(planner_only_cond)),
        GroupAction([rl_node_rl_only], condition=IfCondition(rl_only_cond)),
        GroupAction([rl_node_fm2], condition=IfCondition(fm2_cond)),
        GroupAction([rl_node_sac], condition=IfCondition(sac_cond)),
        GroupAction([rl_node_c2dt05], condition=IfCondition(c2dt05_cond)),
    ])
