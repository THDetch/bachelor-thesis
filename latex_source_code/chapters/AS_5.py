#########################################################################################
# University of Applied Sciences Munich
# Laboratory of Autonomous Systems (LAS)
# moveitpy_jac
# Working with Moveit: Forward kinematics and Jacobian
# schoettl 2024
#########################################################################################

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MoveitPyJac(Node):
    def __init__(self):
        super().__init__('moveitpy_jac')

        self.moveit = MoveItPy(node_name='moveit_py')
        self.planning_scene_monitor = self.moveit.get_planning_scene_monitor()
        self.arm = self.moveit.get_planning_component('arm')
        self.robot_model = self.moveit.get_robot_model()
        self.arm.set_workspace(min_x=-1.0, min_y=-1.0, min_z=0.0, max_x=1.0, max_y=1.0, max_z=2.0)

        self.pub_ = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self.run()

    def build_trajectory(self, joint_angles, times):
        """
        Helper to construct a JointTrajectory message.
        joint_angles: list of joint angle arrays (size 6)
        times: list of times_from_start (seconds)
        """
        msg = JointTrajectory()
        msg.joint_names = ['arm_joint_1', 'arm_joint_2', 'arm_joint_3',
                           'arm_joint_4', 'arm_joint_5', 'arm_joint_6']

        for q, t in zip(joint_angles, times):
            pt = JointTrajectoryPoint()
            if isinstance(q, np.ndarray):
                q = q.tolist()
            pt.positions = q
            pt.time_from_start = Duration(seconds=t).to_msg()
            msg.points.append(pt)

        return msg

    def run(self):
        # 1. get the initial joint angles (for information only)
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            q = robot_state.get_joint_group_positions('arm')
            self.get_logger().info('The initial joint angles are\n' + str(q))

        # 3. perform forward kinematics (for information only)
        pose = robot_state.get_pose(link_name='arm_end_effector')
        self.get_logger().info(f'Pose after executing the initial motion: {pose}')

        dx = np.array([1., 0., 0., 0., 0., 0.]); dz = np.array([0., 0., 1., 0., 0., 0.])
        h = 0.1  # increment factor

        # 4. perform cartesian motion in direction dx
        with self.planning_scene_monitor.read_only() as scene:
            robot_state = scene.current_state
            q1 = robot_state.get_joint_group_positions('arm')
            self.get_logger().info('The current joint angles are\n' + str(q1))

        # get the Jacobian and find an increment dq * h which performs a small step into the desired direction
        J = robot_state.get_jacobian(joint_model_group_name="arm",
                                     reference_point_position=np.array([0.0, 0.0, 0.0]))
        dq = np.linalg.pinv(J) @ dx
        q2 = q1 + dq * h

        robot_state2 = RobotState(self.robot_model)
        robot_state2.set_joint_group_positions('arm', q2)
        robot_state2.update()

        J2 = robot_state2.get_jacobian(joint_model_group_name="arm",
                                       reference_point_position=np.array([0.0, 0.0, 0.0]))
        dq2 = np.linalg.pinv(J2) @ dz
        q3 = q2 + dq2 * h

        robot_state3 = RobotState(self.robot_model)
        robot_state3.set_joint_group_positions('arm', q3)
        robot_state3.update()

        J3 = robot_state3.get_jacobian(joint_model_group_name="arm",
                                       reference_point_position=np.array([0.0, 0.0, 0.0]))
        dq3 = np.linalg.pinv(J3) @ dz
        q4 = q3 - dq3 * h

        robot_state4 = RobotState(self.robot_model)
        robot_state4.set_joint_group_positions('arm', q4)
        robot_state4.update()

        J4 = robot_state4.get_jacobian(joint_model_group_name="arm",
                                       reference_point_position=np.array([0.0, 0.0, 0.0]))
        dq4 = np.linalg.pinv(J4) @ dx
        q5 = q4 - dq4 * h

        robot_state5 = RobotState(self.robot_model)
        robot_state5.set_joint_group_positions('arm', q5)
        robot_state5.update()

        msg = self.build_trajectory([q1, q2, q3, q4, q5], [1., 3., 6., 9., 12.])
        self.pub_.publish(msg)
        time.sleep(10.0)

        # debug info
        self.get_logger().info('robot_state2.joint_positions:\n' + str(robot_state2.joint_positions))

        # remove the following comments to build and publish a shorter message
        # msg = self.build_trajectory([q1, q2, q3, q4], [1., 3., 5., 7.])
        # self.pub_.publish(msg)
        # self.get_logger().info('publishing the message')


def main():
    rclpy.init()
    node = MoveitPyJac()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
