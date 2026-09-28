#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""course_goal_publisher (pure configuration node, Plan A single flow).

Role: publish the final goal point of the selected course (R/L) once to /planner/goal (latched).
Homing is performed directly by rl_offboard_node with a single velocity setpoint stream,
so this node no longer handles position setpoints / OFFBOARD / ARM
(to prevent PX4 OFFBOARD disconnection caused by using two MAVROS plugins simultaneously).

Compatibility: for configurations where rl_offboard_node waits for an external homing_done signal with wait_for_homing=True,
this node publishes /planner/homing_done = True once at startup (the RL node switches its internal phase HOMING/NAVIGATING
by itself, so this signal is auxiliary)."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped


class CourseGoalPublisher(Node):
    """Pure configuration node that publishes the selected course (R/L) goal to /planner/goal (latched).
    Supports runtime course change via /planner/course_select. Frame: odom (ENU relative to spawn)."""

    def __init__(self):
        super().__init__('course_goal_publisher')

        # Params (coordinates/course are overridden by goals.yaml)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('default_course', 'R')
        self.declare_parameter('goal_R', [-23.0, 23.0, 4.0])
        self.declare_parameter('goal_L', [-23.0, 23.0, 4.0])
        self.declare_parameter('spawn_enu', [0.0, 0.0, 0.83])  # = Gazebo spawn (sitl_run.sh)
        # Homing is performed by rl_offboard_node. This node no longer uses the homing parameters,
        # but keeps the declarations for goals.yaml compatibility (no-op).
        self.declare_parameter('enable_homing', True)
        self.declare_parameter('home_xyz', [5.0, 0.0, 4.0])

        # Read
        self.frame_id = self.get_parameter('frame_id').value
        self.course = self.get_parameter('default_course').value
        if self.course not in ('R', 'L'):
            self.get_logger().warn(f'Invalid default_course "{self.course}", using "R"')
            self.course = 'R'

        try:
            goal_r_param = self.get_parameter('goal_R').value
            goal_l_param = self.get_parameter('goal_L').value
            if goal_r_param is None or goal_l_param is None:
                raise ValueError('goal_R/goal_L parameters must be provided via YAML or launch parameters.')
            self.goal_R = self._as_xyz(goal_r_param)
            self.goal_L = self._as_xyz(goal_l_param)
        except (ValueError, TypeError, IndexError) as e:
            self.get_logger().error(f'Failed to parse goal parameters: {e}')
            raise

        self.spawn_enu = self._as_xyz(self.get_parameter('spawn_enu').value)

        # QoS & I/O
        q_latched = QoSProfile(   # goal latching
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_goal = self.create_publisher(PoseStamped, '/planner/goal', q_latched)
        # Compatibility: publish True once in case the RL node subscribes to external homing_done.
        self.pub_homing_done = self.create_publisher(Bool, '/planner/homing_done', q_latched)
        self.sub_select = self.create_subscription(
            String, '/planner/course_select', self._on_select, 10
        )

        self.get_logger().info(
            f'Started course_goal_publisher (config-only) | frame={self.frame_id}, '
            f'default_course={self.course}, goal_R={self.goal_R}, goal_L={self.goal_L}'
        )

        # Publish final goal + homing_done immediately at startup (no delay needed since rl_offboard_node is always doing velocity control).
        self.publish_goal(self.course)
        self._publish_homing_done(True)
        self.get_logger().info('Published initial /planner/goal and /planner/homing_done=True.')

    # ---------- Callbacks ----------
    def _on_select(self, msg: String):
        course = msg.data.strip().upper()
        if course in ('R', 'L'):
            self.course = course
            self.publish_goal(self.course)  # update immediately on course change
        else:
            self.get_logger().warn(f'Unknown course "{course}" (use "R" or "L")')

    # ---------- Helpers ----------
    def _as_xyz(self, arr):
        """Convert to an (x,y,z) tuple (padding if too short)"""
        if not arr or len(arr) == 0:
            raise ValueError('XYZ cannot be empty')
        if len(arr) == 1:
            return float(arr[0]), 0.0, 2.0
        if len(arr) == 2:
            return float(arr[0]), float(arr[1]), 2.0
        return float(arr[0]), float(arr[1]), float(arr[2])

    # ---------- Goal publish ----------
    def publish_goal(self, course: str):
        if course not in ('R', 'L'):
            self.get_logger().warn(f'Unknown course "{course}" (use "R" or "L")')
            return
        gx, gy, gz = self.goal_R if course == 'R' else self.goal_L
        sx, sy, sz = self.spawn_enu
        # goal: (goal - spawn) relative coordinates. In the odom local frame (ENU relative to spawn)
        x, y, z = gx - sx, gy - sy, gz - sz

        goal = PoseStamped()
        goal.header.frame_id = self.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        goal.pose.orientation.w = 1.0

        self.pub_goal.publish(goal)
        self.get_logger().info(f'Published /planner/goal for {course}: ({x:.2f}, {y:.2f}, {z:.2f}) ENU')

    def _publish_homing_done(self, done: bool):
        msg = Bool()
        msg.data = bool(done)
        self.pub_homing_done.publish(msg)


def main():
    rclpy.init()
    node = CourseGoalPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
