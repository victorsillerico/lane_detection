"""Green CAV controller"""

# ************************ EMITTER CAV ****************************

from vehicle import Driver
from controller import GPS, Compass, Emitter, Receiver

import struct
import math 
import numpy as np
import copy

from path_optimizer import PathOptimizer
from math import sin, cos, pi, sqrt

#import scipy.spatial

from lattice_planner import plan_paths, transform_paths, get_closest_index, \
     get_goal_index, collision_check, select_best_path_index

#from pySTL import STLFormula

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

    # communication Link initial configuration
    COMMUNICATION_CHANNEL = 2
    emitter = Emitter("emitter_AV2")
    emitter.setChannel(COMMUNICATION_CHANNEL)

    # verify that the channel is set correctly
    channel = emitter.getChannel();
    if (channel != COMMUNICATION_CHANNEL):
        emitter.setChannel(COMMUNICATION_CHANNEL)

    # linear velocity
    vf = 4
    # distance between front and rear axis in the car
    l = 2

    # global route specification
    global_waypoints = np.array([[0.0, -2.5],[15.0, -2.5], [30.0, -2.5], \
        [45.0, -2.5], [60.0, -2.5]])

    # global route specification REVERSE
    #global_waypoints = np.array([ [60.0, -2.5], [45.0, -2.5], [30.0, -2.5], [15.0, -2.5], [0.0, -2.5] ])
    #global_waypoints = np.array([ [30.0, -2.5], [15.0, -2.5], [0.0, -2.5] ])

    # obstacle's position specification
    obstacles = np.array([ [[37,1.0]], [[37,3]], [[39,3]], [[41,3]] ])

    number_wps = len(global_waypoints)
    
    # waypoint resolution for tracking controller e.g. next-WP[current + delta]
    delta = 3
    
    # reference distance value to enable a new path generation cycle
    threshold =  2
    
    # reference value for planning horizon
    lookahead = 7.0

    # variables for collision checking 
    circle_offsets = [-1.0, 1.0, 3.0] 
    circle_radii = [1.5, 1.5, 1.5]  
    
    # reference value to penalize lattice options
    weight = 10

    # flag variables to active path generation cycles
    enable_path_generation = True
    key = True

    # variable to keep track the number of generated paths
    counter_paths = 0

    # MAIN SIMULATION LOOP 
    while driver.step() != -1:

        # *********** DEFINE CURRENT STATE OF THE EGO-VEHICLE ***********
        # get waypoints to follow: WP[i] and WP[i+delta]

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

        # **************** COMMUNICATION TASKS ****************
        #shares_z_goal = reference_path[len(reference_path)-1][0]
        #shares_x_goal = reference_path[len(reference_path)-1][1]
        message = struct.pack('ff?',coord_ego[0],coord_ego[1],True)
        emitter.send(message)  

        # ********************** GENERATE LATTICES **********************
        
        # get initial and final points for the lattice path
        ego_state = np.array([coord_ego[0], coord_ego[1], ang_heading])

        # get current and next global waypoint
        distances_to_global_wps = np.linalg.norm(global_waypoints - \
            coord_ego,axis=1)

        if key and counter_paths < number_wps-1:
            min_len_to_global_wp, index_min_global_wp = \
                get_closest_index(distances_to_global_wps)
            initial_global_wp = global_waypoints[index_min_global_wp]
            index_next_global = get_goal_index(global_waypoints, lookahead, \
                min_len_to_global_wp, index_min_global_wp)
            next_global_wp = global_waypoints[index_next_global]
            enable_path_generation = True
            key = False
        else:
            pass   
        
        # path generation cycle
        if enable_path_generation:

            vel_ref = 0 # reference value (velocity) --- *REPLACE WITH IDM

            # calculate the goal position (goal_x,goal_y)
            goal_state = np.array([next_global_wp[0],next_global_wp[1],vel_ref])
            goal_state_local = copy.copy(goal_state)
            goal_state_local[0] -= ego_state[0]
            goal_state_local[1] -= ego_state[1]
            goal_x = goal_state_local[0] * np.cos(ego_state[2]) + \
                goal_state_local[1] * np.sin(ego_state[2])
            goal_y = goal_state_local[0] * -np.sin(ego_state[2]) + \
                goal_state_local[1] * np.cos(ego_state[2])

            # calculate the goal heading value
            delta_x = next_global_wp[0] - initial_global_wp[0] 
            delta_y = next_global_wp[1] - initial_global_wp[1] 
            heading_global_path = np.arctan2(delta_y, delta_x)
            goal_t = heading_global_path - ego_state[2]
         
            # keep goal heading within [-pi,pi] so the optimizer behaves well
            if goal_t > pi:
                goal_t -= 2*pi
            elif goal_t < -pi:
                goal_t += 2*pi

            # lattice paths parameters
            path_offset = 6.0 
            num_paths = 3
            goal_state_set = []

            # velocity is preserved after the transformation
            goal_v = 0.5 # constant velocity 

            # get the offset points for the lattice path options
            for i in range(num_paths):
                j = num_paths - (i+1)
                offset = (j - num_paths // 2) * path_offset
                x_offset = offset * np.cos(goal_t + pi/2)
                y_offset = offset * np.sin(goal_t + pi/2)
                goal_state_set.append([goal_x + x_offset, goal_y + y_offset, \
                    goal_t, goal_v])    

            # generate lattice path options
            my_paths, my_path_validity = plan_paths(goal_state_set)

            # transform the paths to match current ego-vehicle configuration
            my_transformed_paths = transform_paths(my_paths, ego_state) 

            # obstacle detection process
            collision_check_array_var = collision_check(my_transformed_paths, \
                obstacles, circle_offsets, circle_radii)

            # get the index for the selected lattice path option
            control_var = select_best_path_index(my_transformed_paths, \
                collision_check_array_var, goal_state, weight)  

            # create the final reference path
            reference_path = \
                np.transpose(my_transformed_paths[control_var][0:2][:])

            #print("**************    NEW PATH GENERATED   *****************")
            counter_paths = counter_paths + 1
            enable_path_generation = False

            # **************** COMMUNICATION TASKS ****************
            #shares_z_goal = reference_path[len(reference_path)-1][0]
            #shares_x_goal = reference_path[len(reference_path)-1][1]
            #message = struct.pack('ff?',coord_ego[0],coord_ego[1],True)
            #emitter.send(message)               

        else:
            pass
       
        # condition to activate a new path generation cycle
        dist_last_wp_on_path = \
            np.linalg.norm(reference_path[len(reference_path)-1] - coord_ego)

        if  dist_last_wp_on_path < threshold:
            key = True
        else:
            key = False   

        # ********************** TRACKING CONTROLLER **********************
        distances_to_path = np.linalg.norm(reference_path - coord_ego,axis=1)

        # find index of minimum value from 2D numpy array
        result_aux = np.where(distances_to_path == np.amin(distances_to_path))
        index_min = result_aux[0][0]
        initial_wp = reference_path[index_min]

        # verify if the current waypoint is the last of the reference path
        if  ( (index_min+delta) > (len(reference_path)-1) ) and \
            counter_paths == number_wps-1:

            next_wp = reference_path[len(reference_path)-1]
            if index_min == len(reference_path)-1:
                vf = 0
            else:
                pass
        else:   # get the next waypoint to track   
            next_wp = reference_path[index_min+delta]

        waypoints = np.array([initial_wp,next_wp])

        # get angle to last waypoint --- waypoint[coordinate_VorH][#wp]
        ang2lastWP = math.atan2(x-waypoints[0][1], z-waypoints[0][0])

        # get angle of the reference path
        angPath = math.atan2(waypoints[1][1]-waypoints[0][1], \
            waypoints[1][0]-waypoints[0][0])
 
        # get crosstrack error
        crosstrack_error = np.amin(np.linalg.norm(coord_ego - waypoints,axis=1))

        # get the crosstrack error sign
        ang_diff = ang2lastWP - angPath

        if ang_diff > np.pi:
            ang_diff -= 2*np.pi
        if ang_diff < -np.pi:
            ang_diff += 2*np.pi

        if ang_diff>0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = -abs(crosstrack_error)

        # get the error in the angle to consider in the control task
        ang_error = ang_heading - angPath

        if ang_error > np.pi:
            ang_error -= 2*np.pi
        if ang_error < -np.pi:
            ang_error += 2*np.pi    

        # parameter for longitudinal control
        k_e = 0.8   

        # get final steering angle
        ang_crosstrack = math.atan2(k_e * crosstrack_error , vf)    
        ro =  ang_error + ang_crosstrack #
        steering_ang = copy.copy(ro)
        steering_ang = min(0.8, steering_ang)
        steering_ang = max(-0.8, steering_ang)
    
        # set final steering angle for the ego-vehicle
        driver.setSteeringAngle(steering_ang)    
        
if __name__ == "__main__":
    main()

