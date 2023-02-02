# **************** METHODS FOR THE LATTICE PLANNER *****************

import numpy as np

from path_optimizer import PathOptimizer
from math import sin, cos, pi, sqrt

import scipy.spatial

def plan_paths(goal_state_set):
    """Plans the path set using the polynomial spiral optimization.
    Plans the path set using polynomial spiral optimization to each of the
    goal states.
    args:
        goal_state_set: Set of goal states (offsetted laterally from one
            another) to be used by the local planner to plan multiple
            proposal paths. These goals are with respect to the vehicle
            frame.
            format: [[x0, y0, t0, v0],
                     [x1, y1, t1, v1],
                     ...
                     [xm, ym, tm, vm]]
            , where m is the total number of goal states
              [x, y, t] are the position and yaw values at each goal
              v is the goal speed at the goal point.
              all units are in m, m/s and radians
    returns:
        paths: A list of optimized spiral paths which satisfies the set of 
            goal states. A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m) along the spiral
                    y_points: List of y values (m) along the spiral
                    t_points: List of yaw values (rad) along the spiral
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
            Note that this path is in the vehicle frame, since the
            optimize_spiral function assumes this to be the case.
        path_validity: List of booleans classifying whether a path is valid
            (true) or not (false) for the local planner to traverse. Each ith
            path_validity corresponds to the ith path in the path list.
    """
    paths         = []
    path_validity = []
    for goal_state in goal_state_set:
        path_optimizer = PathOptimizer()
        path = path_optimizer.optimize_spiral(goal_state[0], 
                                                    goal_state[1], 
                                                    goal_state[2])
        if np.linalg.norm([path[0][-1] - goal_state[0], 
                           path[1][-1] - goal_state[1], 
                           path[2][-1] - goal_state[2]]) > 0.1:
            path_validity.append(False)
        else:
            paths.append(path)
            path_validity.append(True)
    return paths, path_validity

def transform_paths(paths, ego_state):
    """ Converts the to the global coordinate frame.
    Converts the paths from the local (vehicle) coordinate frame to the
    global coordinate frame.
    args:
        paths: A list of paths in the local (vehicle) frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        ego_state: ego state vector for the vehicle, in the global frame.
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
    returns:
        transformed_paths: A list of transformed paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith transformed path, jth point's 
                y value:
                    paths[i][1][j]
    """
    transformed_paths = []
    for path in paths:
        x_transformed = []
        y_transformed = []
        t_transformed = []

        for i in range(len(path[0])):
            x_transformed.append(ego_state[0] + path[0][i]*cos(ego_state[2]) - \
                                                path[1][i]*sin(ego_state[2]))
            y_transformed.append(ego_state[1] + path[0][i]*sin(ego_state[2]) + \
                                                path[1][i]*cos(ego_state[2]))
            t_transformed.append(path[2][i] + ego_state[2])

        transformed_paths.append([x_transformed, y_transformed, t_transformed])

    return transformed_paths


# ************************* EXTRA FUNCTIONS *****************************

def get_closest_index(distances_to_global_wps):
    closest_len = float('Inf')
    closest_index = 0
        
    # Find index of minimum value from 2D numpy array
    
    closest_len = np.amin(distances_to_global_wps)
    
    result_aux = np.where(distances_to_global_wps == closest_len)
    closest_index = result_aux[0][0]             
    # ------------------------------------------------------------------
    return closest_len, closest_index
    
# ************************* GET GOAL INDEX *************************
def get_goal_index(gb_waypoints, lookahead, closest_len, closest_index):
    """Gets the goal index for the vehicle. 
    # Find the farthest point along the path that is within the
    # lookahead distance of the ego vehicle.
    # Take the distance from the ego vehicle to the closest waypoint into
    # consideration.
    """
    arc_length = closest_len
    wp_index = closest_index
    
    # In this case, reaching the closest waypoint is already far enough for
    # the planner.  No need to check additional waypoints.
    if arc_length > lookahead:
        return wp_index
    
    # We are already at the end of the path.
    #if wp_index == len(waypoints) - 1:    #############  REVIEW THIS CONDITION
    #    return wp_index
    
    # Otherwise, find our next waypoint.
    # ------------------------------------------------------------------
    while wp_index < len(gb_waypoints) - 1:
        
        arc_length += np.linalg.norm(gb_waypoints[wp_index+1] - \
            gb_waypoints[wp_index])

        if arc_length >= lookahead:
            wp_index += 1
            break
        else:
            wp_index += 1
    # ------------------------------------------------------------------
    return wp_index

# ********************  OBSTACLE DETECTION ********************
def collision_check(paths, obstacles, circle_offsets, circle_radii):
    """
    Returns a bool array on whether each path is collision free.
    """
    collision_check_array = np.zeros(len(paths), dtype=bool)
    for i in range(len(paths)):
        collision_free = True
        path           = paths[i]
        # Iterate over the points in the path.
        for j in range(len(path[0])):
            # Compute the circle locations along this point in the path.
            # These circle represent an approximate collision
            # border for the vehicle, which will be used to check
            # for any potential collisions along each path with obstacles.
            # The circle offsets are given by self._circle_offsets.
            # The circle offsets need to placed at each point along the path,
            # with the offset rotated by the yaw of the vehicle.
            # Each path is of the form [[x_values], [y_values],
            # [theta_values]], where each of x_values, y_values, and
            # theta_values are in sequential order.
            # Thus, we need to compute:
            # circle_x = point_x + circle_offset*cos(yaw)
            # circle_y = point_y circle_offset*sin(yaw)
            # for each point along the path.
            # point_x is given by path[0][j], and point _y is given by
            # path[1][j]. 
            circle_locations = np.zeros((len(circle_offsets), 2))
            # --------------------------------------------------------------
            circle_locations[:, 0] = [i * int(np.cos(path[2][j])) for i in \
                circle_offsets] + path[0][j]
            circle_locations[:, 1] = [i * int(np.sin(path[2][j])) for i in \
                circle_offsets] + path[1][j]
            # --------------------------------------------------------------
            # Assumes each obstacle is approximated by a collection of
            # points of the form [x, y].
            # Here, we will iterate through the obstacle points, and check
            # if any of the obstacle points lies within any of our circles.
            # If so, then the path will collide with an obstacle and
            # the collision_free flag should be set to false for this flag
            for k in range(len(obstacles)):
                collision_dists = scipy.spatial.distance.cdist(obstacles[k], \
                    circle_locations)

                collision_dists = np.subtract(collision_dists, circle_radii)
                collision_free = collision_free and \
                    not np.any(collision_dists < 0)

                if not collision_free:
                    break
            if not collision_free:
                break
        collision_check_array[i] = collision_free
    return collision_check_array

# *************** BEST PATH INDEX SELECTION ***************
def select_best_path_index(paths, collision_check_array, goal_state, weight):
    """Returns the path index which is best suited for the vehicle to
    traverse.
    Selects a path index which is closest to the center line as well as far
    away from collision paths.
    args:
        paths: A list of paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m)
                    y_points: List of y values (m)
                    t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        collision_check_array: A list of boolean values which classifies
            whether the path is collision-free (true), or not (false). The
            ith index in the collision_check_array list corresponds to the
            ith path in the paths list.
        goal_state: Goal state for the vehicle to reach (centerline goal).
            format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
    useful variables:
        self._weight: Weight that is multiplied to the best index score.
    returns:
        best_index: The path index which is best suited for the vehicle to
            navigate with.
    """
    best_index = None
    best_score = float('Inf')
    for i in range(len(paths)):
        # Handle the case of collision-free paths.
        if collision_check_array[i]:
            # Compute the "distance from centerline" score.
            # The centerline goal is given by goal_state.
            # The exact choice of objective function is up to you.
            # A lower score implies a more suitable path.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # --------------------------------------------------------------
            score = np.sqrt((goal_state[0]-paths[i][0][len(paths[i][0])-1])**2 \
                + (goal_state[1] - paths[i][1][len(paths[i][0])-1])**2) # + robustenss[i]
            # --------------------------------------------------------------
            # Compute the "proximity to other colliding paths" score and
            # add it to the "distance from centerline" score.
            # The exact choice of objective function is up to you.
            for j in range(len(paths)):
                if j == i:
                    continue
                else:
                    if not collision_check_array[j]:
                        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                        # --------------------------------------------------
                        #print("Adding score")
                        score += weight * paths[i][2][j]
                        # --------------------------------------------------
                        pass
        # Handle the case of colliding paths.
        else:
            score = float('Inf')
        #print("score = %f" % score)
        # Set the best index to be the path index with the lowest score
        if score < best_score:
            best_score = score
            best_index = i
    #print("--------------------")
    return best_index



  