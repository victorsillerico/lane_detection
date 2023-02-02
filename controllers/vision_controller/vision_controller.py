"""White CAV controller"""

# ************************ EMITTER CAV ****************************
from vehicle import Driver
from controller import GPS, Compass, Emitter, Receiver, Display, Camera

import struct
import math 
import numpy as np
import copy

def main():
    # create the Vehicle instance
    driver = Driver()

    # get the time step of the current world
    timestep = int(driver.getBasicTimeStep())

    # GPS initial configuration 
    gp = driver.getDevice("global_gps_AV2")
    GPS.enable(gp,timestep)

    # compass initial configuration 
    cp = driver.getDevice("compass_AV2")
    Compass.enable(cp,timestep)

    # Initialize camera
    camera = driver.getDevice('camera')
    Camera.enable(camera,timestep)
    Camera.recognitionEnable(camera,timestep)
    Camera.enableRecognitionSegmentation(camera)
    width = Camera.getWidth(camera)
    height = Camera.getHeight(camera)
    
    display = driver.getDevice('display')
    


    # linear velocity
    vf = 5
    # distance between front and rear axis in the car
    l = 2



    # MAIN SIMULATION LOOP 
    while driver.step() != -1:
        # set longitudinal velocity
        driver.setCruisingSpeed(vf)

        # define the current postion and orientation of the ego-vehicle

        # get GPS values    
        x = gp.getValues()[0]
        y = gp.getValues()[1]
        z = gp.getValues()[2]  

        # get compass values
        comp_x = cp.getValues()[0]
        comp_y = cp.getValues()[1]
        comp_z = cp.getValues()[2]

        # location of ego vehicle
        coord_ego = np.array([z, x])

        # get the heading of the vehicle
        angle2North = math.atan2(comp_x, comp_z) # atan2(Vertical,Horizontal)

        # case I --- angle needs correction  
        if angle2North >= -math.pi and angle2North < -math.pi/2: 
            ang_heading = -3*math.pi/2 - angle2North # HEADING angle
        else:  # cases II, III, IV
            ang_heading = (math.pi/2) - angle2North # HEADING angle          

        #if (Camera.enableRecognitionSegmentation(camera) and Camera.getRecognitionSamplingPeriod(camera)>0 ):
        data = Camera.getRecognitionSegmentationImage(camera)
            
        segmented_image = display.imageNew(data, Display.BGRA, width, height)
        display.imagePaste(segmented_image, 0, 0, blend=False)
        display.imageDelete(segmented_image)
        #print("hello")
  
        # set final steering angle for the ego-vehicle
        driver.setSteeringAngle(0.0)    
        
if __name__ == "__main__":
    main()

