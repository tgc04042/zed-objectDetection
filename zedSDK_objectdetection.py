#!/usr/bin/env python3
import pyzed.sl as sl
import cv2
import numpy as np
import rospy
import math
import statistics
import matplotlib.pyplot as plt
from time import sleep
from std_msgs.msg import Float64MultiArray

width = 1280
height = 720
image_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_global = np.zeros([width, height, 4], dtype=np.float64)
x = []
y = []
plotflag = False

def init_params(zed):
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = True
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

def zed_object_tracking_param(zed):
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking=True
    obj_param.image_sync=True
    obj_param.enable_mask_output=True

    if obj_param.enable_tracking :
        positional_tracking_param = sl.PositionalTrackingParameters()
        #positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS :
        print (repr(err))
        zed.close()
        exit(1)
    return obj_param

def load_image_into_numpy_array(image):
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_depth_into_numpy_array(depth):
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

def plot_boxes_cv2(img, boxes, velocityList, savename=None, class_names=None, color=None):
    global depth_global
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    distance_list = []
    width = img.shape[1]
    height = img.shape[0]
    thickness = 1
    count = 0
    for i in range(len(boxes)):
        box = boxes[i]

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 5 and class_names:
            #for Bounding box
            cls_conf = box[4]
            cls_id = box[5]
            print('%s: %f' % (class_names[count], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            print('class_name:', class_names[count])
            img = cv2.putText(img, class_names[count], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)

            # For object distance
            center_x = (x2 + x1) / 2
            center_y = (y2 + y1) / 2
            box_width = (x2 - x1)
            box_height = (y2 - y1)

            bounds = [center_x, center_y, box_width, box_height]

            x, y, z = get_object_depth(depth_global, bounds)
            distance = math.sqrt(x * x + y * y + z * z)
            distance_list.append(distance)
            distance = "{:.2f}".format(distance)

            cv2.putText(img, str(distance) + " m",
                        (x2 + (thickness * 4), y1 + (thickness * 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            print('velocityList', velocityList)

            cv2.putText(img, str(round(velocityList[i][1],4)) + " m/s",
                        (x2 + (thickness * 4), y2 + (thickness * 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            count += 1

        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)

    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)

    return img, distance_list


def object_tracking(zed, image_np, objects, obj_param, obj_runtime_param):
    err = zed.retrieve_objects(objects, obj_runtime_param)

    global image_global, depth_global

    if objects.is_new:
        obj_array = objects.object_list
        print('\n'+str(len(obj_array)) + " Object(s) detected\n")
        #dimensions_result = np.zeros((len(obj_array), 3))
        objectNum = 0
        #npPosition = np.zeros(len(obj_array))

        box2d = [[0] * 4 for _ in range(len(obj_array))]
        '''
        positionList = [[0] * 3 for _ in range(len(obj_array))]
        dimensionList = [[0] * 3 for _ in range(len(obj_array))]
        velocityList = [[0] * 3 for _ in range(len(obj_array))]
        '''
        positionList = np.array([])
        dimensionList = np.array([])
        velocityList = np.array([])
        velocityList2 = np.array([])
        class_name = []
        if len(obj_array) > 0:
            #print(repr(obj_array))
            for object in obj_array:
                #print('position_list:', positionList)
                object_confidence = int(object.confidence)
                object_label = (object.label)
                object_id = int(object.id)

                print("First object attributes:")
                print(" Label '" + repr(object_label) + "' (conf. " + str(object_confidence) + "/100)")

                if obj_param.enable_tracking:
                    print(" Tracking ID: " + str(object_id) + " tracking state: " + repr(
                        object.tracking_state) + " / " + repr(object.action_state))
                    #class_name[(int(object.id))] = repr(object.label)
                    class_name.append(repr(object.label))

                position = object.position
                velocity = object.velocity
                dimensions = object.dimensions

                positionList = np.append(positionList, np.array(object.position))
                dimensionList = np.append(dimensionList, np.array(object.dimensions))
                velocityList = np.append(velocityList, np.array(object.velocity))
                velocityList2 = np.reshape(np.array(velocityList), (int((len(velocityList) / 3)), 3))

                '''
                positionList[objectNum] = list(map(float, position))
                dimensionList[objectNum] = list(map(float, dimensions))
                velocityList[objectNum] = list(map(float, velocity))
                '''

                #print('posAndDemenList:', posAndDemenList)

                '''
                print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(
                    position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
                    dimensions[1], dimensions[2]))
                '''

                if object.mask.is_init():
                    print(" 2D mask available")

                # bounding box init
                bounding_box_2d = np.array(object.bounding_box_2d)
                bounding_box = object.bounding_box

                coordinate_2d = boundingBox(bounding_box_2d, bounding_box)
                box2d[objectNum] = coordinate_2d
                box2d[objectNum].append(object_confidence)
                box2d[objectNum].append(object_id)

                #distance_list[objectNum] =
                #x, y, z = get_object_depth(dimensions, box2d)
                objectNum+=1
            '''
            print('positionList:', positionList)
            print('dimensionList:', dimensionList)
            print('velocityList:', velocityList)
            '''
        #distance = math.sqrt(x * x + y * y + z * z)
        image_global, distance_list = plot_boxes_cv2(image_np, box2d, velocityList2, class_names=class_name)

        MultiArray.data = positionList
        pub_position.publish(MultiArray)
        MultiArray.data = dimensionList
        pub_dimension.publish(MultiArray)
        MultiArray.data = velocityList
        pub_velocity.publish(MultiArray)
        MultiArray.data = distance_list
        pub_distance.publish(MultiArray)
        #makePyplot(positionList)
        #makePyplot(dimensionList)
        #makePyplot(positionList)

    cv2.imshow('Yolo demo', image_global)
    cv2.waitKey(1)


def makePyplot(li):
    #global x, y
    global plotflag
    count = 0
    x = list()
    y = list()
    for i in li:
        x.append(li[0])
        y.append(li[1])
        print('x:', x)
        print('y:', y)

    if len(x) % 100 == 0:
        plt.plot(x, y, 'ro')
        plt.xlabel('X-Label', fontsize=20)
        plt.ylabel('Y-Label', fontsize=20)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        #plt.pause(0.001)
        #if plotflag is False:
        plt.show()

def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.
    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            #z = depth[i, j]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 1])
                y_vect.append(depth[i, j, 0])
                #x_vect.append(depth[i, j])
                #y_vect.append(depth[i, j])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median

def boundingBox(bounding_box_2d, bounding_box):
    box = []
    for i, it in enumerate(bounding_box_2d):
        if i == 0 or i == 2:
            box.extend(list(map(int, it)))
            print("\nindex :", i, 'value :', it)
        print("    " + str(it), end='')

    print("\n Bounding Box 3D ")
    for it in bounding_box:
        print("    " + str(it), end='')
    return box


def zedCapture(zed):
    width = 1280
    height = 720
    image_np = np.zeros([width, height, 3], dtype=np.uint8)
    image_mat = sl.Mat()
    image_size = sl.Resolution(width, height)

    #zed.retrieve_image(image_mat, sl.VIEW.LEFT, resolution=image_size)
    #image_np_global = load_image_into_numpy_array(image_mat)

def run():
    global image_global, depth_global
    zed = sl.Camera()
    init_params(zed)
    # Create a Camera object

    camera_infos = zed.get_camera_information()

    obj_param = zed_object_tracking_param(zed)
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    objects = sl.Objects()

    width = 1280
    height = 720
    image_np = np.zeros([width, height, 3], dtype=np.uint8)
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    image_size = sl.Resolution(width, height)

    while not rospy.is_shutdown():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT, resolution=image_size)
            zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA, resolution=image_size)
            image_np = load_image_into_numpy_array(image_mat)
            depth_global = load_depth_into_numpy_array(depth_mat)
            object_tracking(zed, image_np, objects, obj_param, obj_runtime_param)
    # Close the camera
    zed.close()

rospy.init_node

if __name__ == "__main__":
    rospy.init_node("object_detection", anonymous=True)
    pub_position = rospy.Publisher('position_data', Float64MultiArray, queue_size=10)
    pub_dimension = rospy.Publisher('dimension_data', Float64MultiArray, queue_size=10)
    pub_velocity = rospy.Publisher('velocity_data', Float64MultiArray, queue_size=10)
    pub_distance = rospy.Publisher('distance_data', Float64MultiArray, queue_size=10)
    MultiArray = Float64MultiArray()
    run()
