import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from tegra_cam import open_cam_onboard

# Utilities for object detector.

import numpy as np
import sys
import os
from threading import Thread
from datetime import datetime
from utils import label_map_util
from collections import defaultdict
from tegra_cam import open_cam_onboard

import math
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil


#Connect to vehicle
print("Connecting...")
vehicle = connect('udp:127.0.0.1:14551')

# Set the commanded flying speed
gnd_speed = 5 # [m/s]

detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def arm_and_takeoff(altitude):

    while not vehicle.is_armable:
        print("waiting to be armable")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed: time.sleep(1)

    print('Taking Off')
    vehicle.simple_takeoff(altitude)

    while True:
        v_alt = vehicle.location.global_relative_frame.alt
        print('>>Altitude = %.lf m'%v_alt)
        if v_alt >= altitude - 1.0:
            print("target altitude reached")
            break
        time.sleep(1)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")
        
    return targetlocation;


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5


def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

#    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        #print "DEBUG: mode: %s" % vehicle.mode.name
 #       remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
  #      print "Distance to target: ", remainingDistance
   #     if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
    #        print "Reached target"
     #       break;

def set_velocity_body(vehicle, vx, vy, vz):
    #vz is positive downward

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111, #-- BITMASK -> Consider only the velocities
            0, 0, 0,        #-- POSITION
            vx, vy, vz,     #-- VELOCITY
            0, 0, 0,        #-- ACCELERATIONS
            0, 0)
    #if vz > 0:
        #break;
    vehicle.send_mavlink(msg)
    vehicle.flush()

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    #global s
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            
            #Detected image center point
            p3 = ((int(left)+int(right))/2)
            p4 = ((int(top)+int(bottom))/2)
            
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            cv2.circle(image_np, (p3, p4), 3, (255, 0, 0), 3)
            
            #Rectangle center points
            x = p3
            y = p4            
            
            #Check quadrant positioning
            if x > image_np.shape[1]/3 and x < image_np.shape[1]*2/3 and y > 0 and y < image_np.shape[0]/3:
                print("UP")
                set_velocity_body(vehicle, 0, 0, -1)

            if x > image_np.shape[1]/3 and x < image_np.shape[1]*2/3 and y > image_np.shape[0]*2/3 and y < image_np.shape[0]:
                print("DOWN")

                set_velocity_body(vehicle, 0, 0, 1)

            if x > 0 and x < image_np.shape[1]/3 and y > image_np.shape[0]/3 and y < image_np.shape[0]*2/3:
                print("RIGHT")
                goto(0, 1)
            if x > image_np.shape[1]*2/3 and x < image_np.shape[1] and y > image_np.shape[0]/3 and y < image_np.shape[0]*2/3:
                print("LEFT") 
                goto(0, -1)
                
            #Up-left, Up-right
            if x > 0 and x < image_np.shape[1]/3 and y > 0 and y < image_np.shape[0]/3:
                print("UP-RIGHT")
                s = "UR"

            if x > image_np.shape[1]*2/3 and x < image_np.shape[1] and y > 0 and y < image_np.shape[0]/3:
                print("UP-LEFT")
                s = "UL"
            
            #Down-left, Down-right
            if x > 0 and x < image_np.shape[1]/3 and y > image_np.shape[0]*2/3 and y < image_np.shape[0]:
                print("DOWN-RIGHT")
                s = "DR"

            if x > image_np.shape[1]*2/3 and x < image_np.shape[1] and y > image_np.shape[0]*2/3 and y < image_np.shape[0]:
                print("DOWN-LEFT")
                s = "DL"
                


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        #self.stream = cv2.VideoCapture(src)
        self.stream = open_cam_onboard(width, height)
        #self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def draw_grid(image_np):
     pxstep = 213
     pystep = 160
     x = pxstep
     y = pystep
     while x < image_np.shape[1]:
         cv2.line(image_np, (x, 0), (x, image_np.shape[0]), (255,255,255), 1, 1)
         x += pxstep
         
     while y < image_np.shape[0]:
         cv2.line(image_np, (0, y), (image_np.shape[1], y), (255,255,255), 1, 1)
         y += pystep


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.2, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-nhands', '--num-hands', dest='num_hands', type=int,
                        default=2, help='Max number of hands to detect.')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default="onboard", help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    #print ("****1")
    detection_graph, sess = load_inference_graph()
    #print ("****2")

    #print ("****4", args.width, args.height)
    if args.video_source == "onboard":
    cap = open_cam_onboard(args.width, args.height)
    else:
        try:
            cap = cv2.VideoCapture(int(args.video_source))
        except ValueError:
            cap = cv2.VideoCapture(args.video_source)
        try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        except Exception as e:
        print("Failed to set frame size")
        print(e.message)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = args.num_hands
    with tf.device('/gpu:0'):
        while True:
            try:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                ret, image_np = cap.read()
                # image_np = cv2.flip(image_np, 1)
                try:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                except:
                    print("Error converting to RGB")
                
                draw_grid(image_np)                
                             
                
                # actual detection
                boxes, scores = detect_objects(
                    image_np, detection_graph, sess)
                    
                # draw bounding boxes
                draw_box_on_image(
                    num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

                #print(detector_utils.s)

                # Calculate Frames per second (FPS)
                num_frames += 1
                elapsed_time = (datetime.datetime.now() -
                                start_time).total_seconds()
                fps = num_frames / elapsed_time

                if (args.display > 0):
                    # Display FPS on frame
                    if (args.fps > 0):
                        draw_fps_on_image(
                            "FPS : " + str(int(fps)), image_np)

                    cv2.imshow('Single Threaded Detection', cv2.cvtColor(
                        image_np, cv2.COLOR_RGB2BGR))

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                else:
                    print("frames processed: ",  num_frames,
                          "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break