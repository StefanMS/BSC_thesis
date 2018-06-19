import numpy as np
import cv2


import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil

import Tkinter as tk


#Connect to vehicle
print("Connecting...")
vehicle = connect('udp:127.0.0.1:14551')

# Set the commanded flying speed
gnd_speed = 5 # [m/s]


#Define arm and takeoff
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

#Define function for sending velocity to body frame
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
    vehicle.send_mavlink(msg)
    vehicle.flush()
    
    
#-- Key event function
def key(event):
    if event.char == event.keysym: #-- standard keys
        if event.keysym == 'r':
            print("r pressed >> Set the vehicle to RTL")
            vehicle.mode = VehicleMode("RTL")
            
    else: #-- non standard keys
        if event.keysym == 'Up':
            set_velocity_body(vehicle, gnd_speed, 0, 0)
        elif event.keysym == 'Down':
            set_velocity_body(vehicle,-gnd_speed, 0, 0)
        elif event.keysym == 'Left':
            set_velocity_body(vehicle, 0, -gnd_speed, 0)
        elif event.keysym == 'Right':
            set_velocity_body(vehicle, 0, gnd_speed, 0)
    
    
#---- MAIN FUNCTION
#- Takeoff
arm_and_takeoff(10)
 
 

face_cascade = cv2.CascadeClassifier('cascade.xml')


cap = cv2.VideoCapture(0)




def draw_grid(img):
    pxstep = 213
    pystep = 160
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), (255,255,255), 1, 1)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), (255,255,255), 1, 1)
        y += pystep



while 1:
    ret, img = cap.read()
    #print(img.shape)
    draw_grid(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 80, 80)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+100,y+100),(0,0,255),2)
        print(x,",",y)
        rectan = cv2.rectangle(img,(x,y),(x+100,y+100),(0,0,255),2)
        cv2.circle(img,(x+50,y+50), 7, (255,0,0), -1)
        x2 = x+50
        y2 = y+50
        if x2 > img.shape[1]/3 and x2 < img.shape[1]*2/3 and y2 > 0 and y2 < img.shape[0]/3:
            print("UP")
            set_velocity_body(vehicle, gnd_speed, 0, 0)

        if x2 > img.shape[1]/3 and x2 < img.shape[1]*2/3 and y2 > img.shape[0]*2/3 and y2 < img.shape[0]:
            print("DOWN")
            set_velocity_body(vehicle, -gnd_speed, 0, 0)

        if x2 > 0 and x2 < img.shape[1]/3 and y2 > img.shape[0]/3 and y2 < img.shape[0]*2/3:
            print("LEFT")
            set_velocity_body(vehicle, 0, -gnd_speed, 0)

        if x2 > img.shape[1]*2/3 and x2 < img.shape[1] and y2 > img.shape[0]/3 and y2 < img.shape[0]*2/3:
            print("RIGHT") 
            set_velocity_body(vehicle, 0, gnd_speed, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Hand',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
