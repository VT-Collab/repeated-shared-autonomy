import rospy
import sys
import time
import numpy as np


from std_msgs.msg import Float32MultiArray

# remote teleop imports
import firebase_admin
from firebase_admin import db, credentials

cred = credentials.Certificate('pk.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rsa-remote-op-default-rtdb.firebaseio.com/'})

class FirebaseClient(object):
    def __init__(self):
        self.ref = db.reference('coords/')
        self.firebase_pub = rospy.Publisher('/remote_cmd', Float32MultiArray, queue_size=2)

    def pub_action(self):
        inputs = self.ref.get()
        axes = np.array([inputs["x"], inputs["y"], inputs["z"], inputs["rx"], inputs["ry"], inputs["rz"]])
        grip = inputs["grip"]
        start = inputs["stop"]
        slow = inputs["slow"]
        mode = 0
        assist = inputs["assist"]
        remote_data = Float32MultiArray()
        remote_data.data = [inputs["x"] * 0.25, inputs["y"] * 0.25, inputs["z"] * 0.25, \
            inputs["rx"] * 0.25, inputs["ry"] * 0.25, inputs["rz"] * 0.25,\
            mode, grip, slow, start, assist]
        self.firebase_pub.publish(remote_data)

def main():
    firebase = FirebaseClient()
    rospy.init_node("remote_publisher")
    rate = rospy.Rate(1000)

    print("[*] Initialized, publishing action commands")

    while not rospy.is_shutdown():
        firebase.pub_action()
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

