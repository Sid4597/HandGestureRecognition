#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import time
from RobotClass import Robot
from collections import defaultdict

robot = Robot()


def robotcontrol(command):
    robot_dict = {
    "forward":robot.forward(0.5),
    "backward": robot.backward(0.5),
    "left": robot.left(0.5),
    "right": robot.right(0.5),
    "spin": robot.left(1),
    "stop":robot.stop()
    }
    # if command == 'forward':
    #     robot.forward(0.5)
    # elif command == 'backward':
    #     robot.backward(0.5)
    # elif command == 'left':
    #     robot.left(0.5)
    # elif command == 'right':
    #     robot.right(0.5)
    # elif command == 'spin':
    #     robot.left(0.5)
    #     robot.right(0.5)
    # elif command == 'stop':
    #     robot.stop()
    # else:
    #     robot.stop()

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Command Received: ", data.data)
    robotcontrol(data.data)
def listener():
    rospy.init_node('rebotcontroller', anonymous=True)
    rospy.Subscriber("commands", String, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
