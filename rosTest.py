import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np
import pyqt_test as pt
import sys
import message_filters
import math

'''
def pyqtTest():
    app = pt.QApplication(sys.argv)
    ex = pt.MyApp()
    sys.exit(app.exec_())
'''

def run(posData, dimData, velData, disData):
    global posionList, dimensionList, velocityList, distanceList
    positionList = np.reshape(np.array(posData.data), (int((len(posData.data) / 3)), 3))
    dimensionList = np.reshape(np.array(dimData.data), (int((len(dimData.data) / 3)), 3))
    velocityList = np.reshape(np.array(velData.data), (int((len(velData.data) / 3)), 3))
    distanceList = np.array(disData.data)

    print('positionList:', positionList)
    print('dimensionList:', dimensionList)
    print('velocityList:', velocityList)
    print('distanceList:', distanceList)

    speedList = objectSpeedAndDirection
    objectAvoidance(positionList, dimensionList, distanceList)

def myCarInformation():
    camera_to_vehicle_length = 0.05
    return camera_to_vehicle_length

def objectSpeedAndDirection(velocityList):
    speedList = []
    for vel in velocityList:
        speedList.append(math.sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]))

    return speedList

def objectAvoidance(positionList, dimensionList, distanceList):
    count = 0
    for pos in positionList:
        if count == len(pos):
            count = 0
        #if pos[0] ==



'''
def pyqtTest():
    global positionList
    app = pt.QApplication(sys.argv)
    form = pt.QDialog()

    button = pt.QPushButton("Quit")
    button.clicked.connect(pt.QCoreApplication.instance().quit)
    while not rospy.is_shutdown():
        map = pt.MapUI(positionList)
        layout = pt.QVBoxLayout()
        layout.addWidget(map)
        layout.addWidget(button, stretch=0)
        form.setLayout(layout)

        form.setWindowTitle("particle filter Localization")
        form.show()
        app.exec_()
'''


if __name__ == "__main__":
    rospy.init_node('my_node', anonymous=True)
    sub_Position = message_filters.Subscriber('position_data', Float64MultiArray)
    sub_dimension = message_filters.Subscriber('dimension_data', Float64MultiArray)
    sub_velocity = message_filters.Subscriber('velocity_data', Float64MultiArray)
    sub_distance = message_filters.Subscriber('distance_data', Float64MultiArray)
    ts = message_filters.ApproximateTimeSynchronizer(
        [sub_Position, sub_dimension, sub_velocity, sub_distance], 10,
        0.1, allow_headerless=True)
    ts.registerCallback(run)
    #pyqtTest()

    rospy.spin()