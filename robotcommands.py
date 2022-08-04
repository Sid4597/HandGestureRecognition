#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import time
from RobotClass import Robot

robot = Robot()

def robotcontrol(command):
    if command != "":
        if command == 'forward':
            robot.forward(0.5)






def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Command Received: ", data.data)
    robotcontrol(data.data)
def listener():
    rospy.init_node('rebotcontroller', anonymous=True)
    rospy.Subscriber("commands", String, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
