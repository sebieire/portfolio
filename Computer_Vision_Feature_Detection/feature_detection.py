"""
Machine Vision - Feature Point Detection and Image Matching
Implementation of SIFT-inspired feature detection and correlation-based matching
"""

#########################################
        # IMPORTS #
#########################################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plots
from collections import Counter
import cv2
import copy


#########################################
        # GLOBAL FLAGS #
#########################################

# --------------------------
# --- RUN WHICH TASK(s) ---
# --------------------------

RUN_TASKS = [1] # include 1 and/or 2: [1], [2], [1,2]

# --------------------------
# --- TASK 1 FLAGS ---
# --------------------------
# 1A
OUTPUT_1A_IMG = False # False = default (no images shown)

# 1B
OUTPUT_1B_PLOT = True
OUTPUT_1B_IMG = True
SIGMA_DEPENDANT_KERNEL = True # True = default (important!)

# 1C
OUTPUT_1C_IMG = True

# 1D
EXECUTE_1D = True # takes a while to run (can be switched off when not needed - backup is in place!)
OUTPUT_1D_SUMMARY = True # summary only (recommended!)
OUTPUT_1D_POINTS = False # displays full list of points 
OUTPUT_1D_DEBUG = False # careful - will output every step! = CONSOLE OVERLOAD!

# 1E
OUTPUT_1E_IMG = True

# 1F
OUTPUT_1F_CONSOLE = False

# 1E
OUTPUT_1E_IMG = True

# --------------------------
# --- TASK 2 FLAGS ---
# --------------------------
# 2A
OUTPUT_2A_IMG = True

# 2B
OUTPUT_2B_IMG = True

# 2C
EXECUTE_2C = True # takes a while to run (can be switched off when not needed)
OUTPUT_2C_IMG = True
OUTPUT_2C_CONSOLE = False



#########################################
        # TASK 1 FUNCTIONS #
#########################################

def process_input_image_t1a(input_img):
    """        
    @description: task 1A / process input image into grey scale and resize
    @return: grey scale image, scaled grey scale image
    """
    
    print("- Running Task 1A -")

    # convert input img to grey
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)    
    
    # convert to float
    img = img.astype(np.float32) / np.max(img)
    
    # resize img
    original_height = img.shape[0]  
    original_width = img.shape[1]
    
    img_resized = cv2.resize(img, (original_width*2,original_height*2))
    
    # optional output
    if OUTPUT_1A_IMG:
        cv2.imshow("Gray Scale Img", img)
        cv2.imshow("Gray Scale Img Resized", img_resized)
    
    # return results
    return img, img_resized


def gaussian_smoothing_kernels_t1b(input_img, k_values, sigma_dependant_scale_plot=True):
    """        
    @description: task 1B
    @param: "sigma_dependant_scale_plot" should be used when applying kernels (default)
        set to False ONLY when a clearer visualisation (difference of kernels) is desired
        as this will otherwise return different kernel results (sigma idependant: factor 20!)
    @return: map (images dictionary)
    """
    
    print("- Running Task 1B -")
    
    image_map = {}

    # create 12 kernels and plot
    for k in k_values:
        
        # calculate sigma value
        sigma = 2 ** (k/2)
    
        # meshgrid: sigma (-3 to 3 inclusive)
        if sigma_dependant_scale_plot:
            x,y = np.meshgrid( np.arange(-3*sigma, 3*sigma+1), np.arange(-3*sigma, 3*sigma+1))        
        else:
            # use this ONLY for additional kernel visualisation (static scale for all) and NOT for computation of kernel on images
            x,y = np.meshgrid( np.arange(-20, 20, 0.1), np.arange(-20, 20, 0.1))                
        
        # calculate kernel     
        gauss_smooth_kernel = (1 / 2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))
        
        # apply kernel to resized img
        filt_img = cv2.filter2D(input_img, -1, gauss_smooth_kernel)
        filt_img = filt_img/np.max(filt_img)        
        
        # save to map
        image_map[k] = filt_img
 
        # output: show kernels (plots)
        if OUTPUT_1B_PLOT:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
            ax.plot_surface(x, y, gauss_smooth_kernel)            
            
    # output: show images 
    if OUTPUT_1B_IMG:        
        # initial placeholder only
        v_stack = np.zeros(len(input_img[0])*2)
        
        for i in range(0,len(image_map)):            
            # every even: merge 2 images horizontally
            if i%2 == 0:
                h_stack = np.hstack((image_map[i],image_map[i+1]))
            # every other: merge vertically
            if i%2 == 1:
                v_stack = np.vstack((v_stack,h_stack))
                
        cv2.namedWindow("1B Resulting Images", cv2.WINDOW_NORMAL)
        # reduce window size for displaying
        cv2.resizeWindow("1B Resulting Images", 682, 1536)
        cv2.imshow("1B Resulting Images", v_stack)
    
    # return results            
    return image_map
    

def difference_of_gaussian_t1c(image_map):
    """        
    @description: task 1C / display images of DoG
    @return: map of difference of gaussians (length of 11)
    """
    
    print("- Running Task 1C -")
    
    DoG_map = {}
    
    # calculate difference of gaussian 
    for i in range(0, len(image_map)-1):        
        DoG_map[i] = image_map[i+1] - image_map[i]        
    
    # output: show DoG as images 
    if OUTPUT_1C_IMG:        
        # initial placeholder
        v_stack = np.zeros(len(image_map[0][0])*2)        
        
        for i in range(0,len(DoG_map)-1):
            # every even: merge 2 images horizontally
            if i%2 == 0:
                h_stack = np.hstack((DoG_map[i],DoG_map[i+1]))
            # every other: merge vertically
            if i%2 == 1:
                v_stack = np.vstack((v_stack,h_stack))
        
        # add last image (plus empty slot) as number of img is uneven
        empty_slot = np.zeros((1536,2048))
        h_stack = np.hstack((DoG_map[10],empty_slot))
        v_stack = np.vstack((v_stack,h_stack))
                            
        cv2.namedWindow("1C DoG Resulting Images", cv2.WINDOW_NORMAL)
        # reduce window size for displaying
        cv2.resizeWindow("1C DoG Resulting Images", 682, 1536)
        cv2.imshow("1C DoG Resulting Images", v_stack)
        
    # return results      
    return DoG_map

    
def keypoints_non_maxima_suppression_t1d(DoG_map, threshold):
    """        
    @description: task 1D
    if this function is not run there is a small "backup" array in place 
    @return: keypoints array
    """
    
    print("- Running Task 1D -")
    print("(Please be patient - This takes a bit of time to compute)")
    
    keypoints = []
    
    # go through each DoG map (1 to -1 = length of 9)
    for scale in range(1,len(DoG_map)-1):
        # go through each row (leave 1 gap on each side)
        for row in range(1,len(DoG_map[scale])-1):
            # go through each column (leave 1 gap on each side)
            for col in range(1, len(DoG_map[scale][0])-1):
                
                # go through each point on current scale (level)
                current_interest_point = DoG_map[scale][row,col]
                
                # go through each point on this scale (level)
                if (current_interest_point*255 > threshold and 
                    current_interest_point > DoG_map[scale][row,col-1] and #left
                    current_interest_point > DoG_map[scale][row-1,col-1] and #top left
                    current_interest_point > DoG_map[scale][row-1,col] and # top
                    current_interest_point > DoG_map[scale][row-1,col+1] and #top right
                    current_interest_point > DoG_map[scale][row,col+1] and #right
                    current_interest_point > DoG_map[scale][row+1,col+1] and #btm right
                    current_interest_point > DoG_map[scale][row+1,col] and #btm
                    current_interest_point > DoG_map[scale][row+1,col-1] and #btm left
                    
                    # scale below
                    current_interest_point > DoG_map[scale-1][row,col-1] and
                    current_interest_point > DoG_map[scale-1][row-1,col-1] and
                    current_interest_point > DoG_map[scale-1][row-1,col] and
                    current_interest_point > DoG_map[scale-1][row-1,col+1] and
                    current_interest_point > DoG_map[scale-1][row,col+1] and
                    current_interest_point > DoG_map[scale-1][row+1,col+1] and
                    current_interest_point > DoG_map[scale-1][row+1,col] and
                    current_interest_point > DoG_map[scale-1][row+1,col-1] and
                    current_interest_point > DoG_map[scale-1][row,col] and
                    
                    # scale above
                    current_interest_point > DoG_map[scale+1][row,col-1] and
                    current_interest_point > DoG_map[scale+1][row-1,col-1] and
                    current_interest_point > DoG_map[scale+1][row-1,col] and
                    current_interest_point > DoG_map[scale+1][row-1,col+1] and
                    current_interest_point > DoG_map[scale+1][row,col+1] and
                    current_interest_point > DoG_map[scale+1][row+1,col+1] and
                    current_interest_point > DoG_map[scale+1][row+1,col] and
                    current_interest_point > DoG_map[scale+1][row+1,col-1] and
                    current_interest_point > DoG_map[scale+1][row,col]):
                    
                    # NOTE: scale here symbolises the original k as used in task 1B
                    # the real sigma value can be derived from that (position in k length sigma values array)
                    keypoints.append([row,col,scale+1]) # +1 is the correct k scale value
                
                    
        if OUTPUT_1D_DEBUG:
            print("--------------")
            print("Current Keypoints List:\n", keypoints)
            print("Current Scale (Level):", scale)
            print("Keypoints Length:", len(keypoints))
            print("--------------\n")    

    if OUTPUT_1D_POINTS:
        print("All Keypoints\n:", keypoints)
    
    if OUTPUT_1D_SUMMARY:
        print("Keypoints Length:\n", len(keypoints))
    
    # return results
    return keypoints


def calcualte_derivatives_t1e(image_map):
    """        
    @description: task 1E
    @return: derivative images (dx,dy) maps
    """
    
    print("- Running Task 1E -")
    
    dx_map = {} # 12
    dy_map = {} # 12
    
    kernel_x = np.array([[1,0,-1]])
    kernel_y = np.array([[1,0,-1]]).T

    for key in image_map:
        derivative_x = cv2.filter2D(image_map[key],-1,kernel_x)
        derivative_y = cv2.filter2D(image_map[key],-1,kernel_y)
        
        # match keys of img_map with the derivative mapped keys (0-11)
        dx_map[key] = derivative_x/np.max(derivative_x)
        dy_map[key] = derivative_y/np.max(derivative_y)
        
    # output: show derivatives as images
    if OUTPUT_1E_IMG:        
        # initial placeholder
        v_stack_dx = np.zeros(len(dx_map[0][0])*2)
        v_stack_dy = np.zeros(len(dy_map[0][0])*2)
        
        for i in range(0,len(image_map)):
            # every even: merge 2 images horizontally
            if i%2 == 0:
                h_stack_dx = np.hstack((dx_map[i],dx_map[i+1]))
                h_stack_dy = np.hstack((dy_map[i],dy_map[i+1]))
            # every other: merge vertically
            if i%2 == 1:
                v_stack_dx = np.vstack((v_stack_dx,h_stack_dx))
                v_stack_dy = np.vstack((v_stack_dy,h_stack_dy))
        
        # display & reduce window size for displaying        
        cv2.namedWindow("1E Resulting Derivatives dx", cv2.WINDOW_NORMAL)        
        cv2.resizeWindow("1E Resulting Derivatives dx", 682, 1536)
        cv2.imshow("1E Resulting Derivatives dx", v_stack_dx)
        
        cv2.namedWindow("1E Resulting Derivatives dy", cv2.WINDOW_NORMAL)        
        cv2.resizeWindow("1E Resulting Derivatives dy", 682, 1536)
        cv2.imshow("1E Resulting Derivatives dy", v_stack_dy)
    
    # return results
    return dx_map, dy_map


def calcualte_gradient_length_direction_weights_t1f_1(key_points_list, horizontal_gradient_map, vertical_gradient_map):
    """        
    @description: task 1F - part 1
    @return: gradient_magnitudes, gradient_directions, weights
    """
    
    print("- Running Task 1F (part 1) -")
    
    # VARS    
    max_rows = len(horizontal_gradient_map[0])
    max_cols= len(horizontal_gradient_map[0][0])
    gradient_magnitudes = {}
    gradient_directions = {}
    weights = {}    
    
    for keypoint in key_points_list:
        # sigma value from each keypoint
        sigma = keypoint[2]
        # columns,row meshgrid (matrices) of sigma * (-3,..,3) "modifiers"
        cols,rows = np.meshgrid( np.arange(-3, 4)* 3/2 * sigma, np.arange(-3, 4) * 3/2 * sigma)
        
        # calculate actual points coordinates (cols & rows) according to modifiers 
        # flat arrays with same length
        rows = (rows + keypoint[0]).flatten()
        cols = (cols + keypoint[1]).flatten()
        
        # dicts with (empty) nparrays
        gradient_magnitudes[(keypoint[0],keypoint[1])] = np.empty([0,1], dtype=float)
        gradient_directions[(keypoint[0],keypoint[1])] = np.empty([0,1], dtype=float)
        weights[(keypoint[0],keypoint[1])] = np.empty([0,1], dtype=float)
        
        # calculate magnitutes, directions & weights
        for i in range(len(rows)): # same as len(cols)            

            # - - - Get Correct gx & gy - - -
            # IMPORTANT: deal with potential out of bounds issues
            if rows[i] < 0 or rows[i] >= max_rows or cols[i] < 0 or cols[i] >= max_cols:
                # if out of bound use the actual keypoint instead
                gx = horizontal_gradient_map[sigma][keypoint[0]][keypoint[1]] # the actual keypoint
                gy = vertical_gradient_map[sigma][keypoint[0]][keypoint[1]]
                
            # regular case (not out of bounds)
            else:
                gx = horizontal_gradient_map[sigma][int(rows[i])][int(cols[i])] # the current grid point
                gy = vertical_gradient_map[sigma][int(rows[i])][int(cols[i])]                 
            
            # - - - Magnitutes, Directions & Weights - - -            
            gradient_magnitudes[(keypoint[0],keypoint[1])] = np.append(gradient_magnitudes[(keypoint[0],keypoint[1])] , np.sqrt(gx**2 + gy**2))  # append to mapped list                
            gradient_directions[(keypoint[0],keypoint[1])] = np.append(gradient_directions[(keypoint[0],keypoint[1])], np.arctan2(gy,gx))
            
            #wqr = np.exp( -(rows[i]**2 + cols[i]**2) / (9 * sigma**2 / 2) ) / (9 * np.pi * sigma **2 / 2)
            wqr = np.exp( -(gx**2 + gy**2) / (9 * sigma**2 / 2) ) / (9 * np.pi * sigma **2 / 2) # weight function            
            weights[(keypoint[0],keypoint[1])] = np.append(weights[(keypoint[0],keypoint[1])], wqr )

    # Diagnostic
    if OUTPUT_1F_CONSOLE:
        print("No. Keypoints :", len(key_points_list))
        print("No. Gradient Magnitude Maps (a 49 magnitudes each):", len(gradient_magnitudes))
        print("No. Gradient Direction Maps (a 49 directions each):", len(gradient_directions))
        print("No. Weights Direction Maps (a 49 weights each):", len(weights))
    
    # return results
    return gradient_magnitudes, gradient_directions, weights
    

def histogram_t1f_2(gradient_magnitudes_map, gradient_directions_map, weights_map):
    """
    @description: task 1F - part 2
    NOTE: all function parameters are identical in length and have the same keys for mapping
    @return: keypoint_orientations
    """
    
    print("- Running Task 1F (part 2) -")
    
    keypoint_orientations = {}
    weighted_gradient_magnitudes = {}
    
    for keypoint in gradient_magnitudes_map: 
        
        keypoint_orientations[keypoint] = np.empty([0,1], dtype=float)
        
        # - - - - - - - -
        # find max minus min values
        minValue = np.min(gradient_directions_map[keypoint])
        maxValue = np.max(gradient_directions_map[keypoint])        
        
        # - - - - - - - -
        # define 1 of 36 steps
        oneStepOf36 = np.abs(maxValue - minValue) / 36
        # values split into 36 subdivision (create a 36-bin vector from theta (gradient directions min/max))
        histogram_direction_values_bin = np.arange(minValue, maxValue+0.01, oneStepOf36, dtype=float) # add to max so that last value is included
        binCounter = np.zeros(len(histogram_direction_values_bin))
        
        # weighted gradients lengths
        weighted_gradient_magnitudes[keypoint] = gradient_magnitudes_map[keypoint] * weights_map[keypoint]
        
        # cycle each gradient of this keypoint and "assign" over 36 bins (for hit = +1 * weighted of that same grid point)
        for i in range(0, len(gradient_directions_map[keypoint])):
            slot = np.sum(gradient_directions_map[keypoint][i] > histogram_direction_values_bin ) - 1
            binCounter[slot] += 1 * weighted_gradient_magnitudes[keypoint][i] # using 1 here but weighted each time!
            
        # take bin with highest frequency (= keypoint orientation)
        keypoint_orientations[keypoint] = histogram_direction_values_bin[np.argmax(binCounter)]                
    
    # return results
    return keypoint_orientations
    

def draw_keypoints_and_output_img_t1g(img, keypoints,keypoint_orientations):
    """
    @description: task 1G   
    @return: void  
    """
    
    print("- Running Task 1G -")
        
    for keypoint in keypoints:
        
        theta = keypoint_orientations[(keypoint[0],keypoint[1])]
        sigma = keypoint[2]
        point_a = (int(keypoint[1]/2),int(keypoint[0]/2))
        point_b =  (int((keypoint[1]/2) + 3 * sigma * np.cos(theta)), int((keypoint[0]/2) + 3 * sigma * np.sin(theta)))
        
        image = cv2.circle(img, point_a, 3 * keypoint[2], [0,0,255], 1)        
        image = cv2.line(img, point_a, point_b, [0,0,255], 1 )
        
    cv2.imshow("Task 1E - Keypoint Results", image)
    
    
    

#########################################
        # TASK 2 FUNCTIONS #
#########################################

def process_input_images_t2a(input_img_1, input_img_2):
    """
    @description: task 2A / process input images into grey scale
    @return: 2 grey scale images
    """
    
    print("- Running Task 2A -")

    # convert input images to grey
    img1 = cv2.cvtColor(input_img_1, cv2.COLOR_RGB2GRAY)   
    img2 = cv2.cvtColor(input_img_2, cv2.COLOR_RGB2GRAY)
    
    # convert to float
    img1 = img1.astype(np.float32) / np.max(img1)
    img2 = img2.astype(np.float32) / np.max(img2)
    
    # optional output
    if OUTPUT_2A_IMG:
        cv2.imshow("Task 2A - Gray Scale Img 1", img1)
        cv2.imshow("Task 2A - Gray Scale Img 2", img2)
    
    # return results
    return img1, img2


def draw_and_cut_elements_t2b(img, point_a, point_b):
    """        
    @description: task 2B rectangle and window cutout
    @return: image patch
    """     
    
    print("- Running Task 2B -")
    
    # rectangle around window
    img_rect = copy.copy(img)
    
    img_rect = cv2.rectangle(img_rect, point_a, point_b, (0,0,0), 2)
    
    # cut image
    cut_height = np.abs(point_a[0] - point_b[0])
    cut_width = np.abs(point_a[1] - point_b[1])
    cropped_img = img[point_a[1]:point_a[1] + cut_width, point_a[0]:point_a[0] + cut_height]
    
    if OUTPUT_2B_IMG:
        cv2.imshow("Task 2B - Rectangle Around Window", img_rect)
        plt.imshow(cropped_img)
       
    # return results
    return cropped_img


def cross_correlation_t2c(img_patch, img2):
    """
    @description: task 2C cross correlation
    @return: void
    """
    
    print("- Running Task 2C -")
    print("(Please be patient - This takes a bit of time to compute)")
    
    # mean & std deviation of patch
    patch_mean_deviation = img_patch - np.mean(img_patch)
    patch_std_deviation = np.std(img_patch)    
    
    # setup range variables
    patch_height = img_patch.shape[0]
    patch_width = img_patch.shape[1]
    max_row = img2.shape[0] - patch_height + 1
    max_col = img2.shape[1] - patch_width + 1    
     
    p = {}
    
    # go through all positions in img2
    for row in range(0,max_row):
        for col in range(0, max_col):
            # compare patch
            compare_patch = img2[row:row+patch_height, col:col+patch_width]
            
            # calculate mean & std deviation of comparison patch            
            compare_mean_deviation = compare_patch - np.mean(compare_patch)
            compare_std_deviation = np.std(compare_patch)            
        
            # cross correlation            
            p[row,col] = np.sum(patch_mean_deviation * compare_mean_deviation) / len(img_patch.flatten()) / (patch_std_deviation * compare_std_deviation)
            
    # key of max value    
    max_p_value_key = max(p, key=p.get)
    
    # output: show images output
    if OUTPUT_2C_IMG:
        
        # - - DISPLAY RANGE OF IMAGES (Merged) - -
        # initial placeholders only
        v_stack = np.zeros(patch_width*10 + 1) # +1 for 1 pixel extra in first h_stack
        h_stack = np.zeros((patch_height,1)) # 1 pixel column placeholder
        # 10 images per row counter
        ten_per_row_counter = 0
        
        # TOP 50 SOLUTIONS
        # there are about 65 available with "relatively" correct target patch (using only first 50 here)
        top_50_solutions = dict(Counter(p).most_common(50))
        
        for key in top_50_solutions:
            
            x_coord = key[0]
            y_coord = key[1]
            solution_patch = img2[x_coord:x_coord+patch_height, y_coord:y_coord+patch_width]
            
            h_stack = np.hstack((h_stack,solution_patch))            
            ten_per_row_counter += 1
            
            if ten_per_row_counter == 10:
                
                # add row to v_stack
                v_stack = np.vstack((v_stack,h_stack))
                # new row (h stack)
                h_stack = np.zeros((patch_height,1))                
                #reset counter for next row
                ten_per_row_counter = 0
            
        cv2.namedWindow("Task 2C - Top 50 Potential Positions", cv2.WINDOW_NORMAL)
        # reduce window size for displaying
        cv2.resizeWindow("Task 2C - Top 50 Potential Positions", 700, 450)
        cv2.imshow("Task 2C - Top 50 Potential Positions", v_stack)
        
        # - - DISPLAY MAXIMUM CORRELATION RECTANGLE ON IMG - -
        point_a = (max_p_value_key[1],max_p_value_key[0])
        point_b = (max_p_value_key[1]+patch_width, max_p_value_key[0]+patch_height)        
        img_rect = cv2.rectangle(img2, point_a, point_b, (0,0,0), 2)
        cv2.imshow("Task 2C - Maximum Cross Correlation Rectangle", img_rect)
    
    if OUTPUT_2C_CONSOLE:
        print("Top Potential Solutions (max values):\n", top_50_solutions)
        print("Max P (row,col):", max_p_value_key)
        print("Max P value:", p[max_p_value_key])
        


#########################################
        # RUN #
#########################################
    
def main():
    
    # TASK 1
    if 1 in RUN_TASKS:
        
        print("\n- - RUNNING TASK 1 - -\n")
        
        ##################
        # TASK 1A
        input_img = cv2.imread("sample_image_1.jpg")    
        img, img_resized = process_input_image_t1a(input_img)    
            
        ##################
        # TASK 1B
        k_values = np.arange(0,12)
        img_map = gaussian_smoothing_kernels_t1b(img_resized, k_values, SIGMA_DEPENDANT_KERNEL)        
        
        ##################
        # TASK 1C
        DoG_map = difference_of_gaussian_t1c(img_map)    
        
        ##################
        # TASK 1D
        if EXECUTE_1D:
            threshold = 10
            keypoints = keypoints_non_maxima_suppression_t1d(DoG_map, threshold)
        else: # backup scenario (very small keypoints array for quick testing only)
            keypoints = [[601, 1125, 2], [601, 1195, 2], [31, 301, 6], [111, 76, 9], [844, 1122, 10]]
        
        ##################
        # TASK 1E    
        derivatives_x, derivatives_y = calcualte_derivatives_t1e(img_map)
        
        ##################
        # TASK 1F
        
        # - part 1 -
        gradient_magnitudes, gradient_directions, weights = calcualte_gradient_length_direction_weights_t1f_1(keypoints, derivatives_x, derivatives_y)        
        
        # - part 2 -
        keypoint_orientations = histogram_t1f_2(gradient_magnitudes, gradient_directions, weights)
        
        ##################
        # TASK 1G
        draw_keypoints_and_output_img_t1g(input_img,keypoints,keypoint_orientations)
        
    
    # TASK 2
    if 2 in RUN_TASKS:
        
        print("\n- - RUNNING TASK 2 - -\n")        
        
        ##################
        # TASK 2A
        input_img_1 = cv2.imread("sample_image_1.jpg")
        input_img_2 = cv2.imread("sample_image_2.jpg")
        img1, img2 = process_input_images_t2a(input_img_1,input_img_2)
    
        ##################
        # TASK 2B
        point_a = (360,210)
        point_b = (430,300)        
        
        img_patch = draw_and_cut_elements_t2b(img1, point_a, point_b)
        
        ##################
        # TASK 2C
        if EXECUTE_2C:            
            cross_correlation_t2c(img_patch, img2)
    
    
    ##################
    # wait for key & close
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
    
    
    