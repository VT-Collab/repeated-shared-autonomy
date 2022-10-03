import rospy
from utils import TrajectoryClient

def main():
    rospy.init_node("cart_pos")
    mover= TrajectoryClient()

    while not rospy.is_shutdown():
        pos = mover.joint2pose()
        print(pos[3:])


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass