import tkinter
import numpy as np
global T_buffer
def initialize_status_bar(root):
    global T_buffer
    T_buffer = tkinter.Label(root, height=1, width=60)
    T_buffer.pack(side=tkinter.BOTTOM)

def tkinter_display(message):
    #tkinter.Label(T_buffer, text=message).pack()
    T_buffer.config(text=message)
    T_buffer.update()


# ----------- Parameters Specifications ----------- #

ratio = 4
gaussian_anti_aliasing_blur_size = (3,3)
gaussian_anti_aliasing_blur_coef = 0

# --- Stabilization Parameters --- #

maxCorners = 400
qualityLevel = 0.0001
minDistance = 70

video_borders_coef = 1.06
smoothness_coef = 15

# --- Background Subtraction Parameters --- #

median_video_kernel = 31
median_frame_gaussian_blur_size = (3,3)
median_frame_gaussian_blur_sigma = 80
binary_frame_threshold = 150
binary_frame_eroding_size = (9, 9)
binary_frame_eroding_iterations = 1
binary_frame_dilating_size = (13, 13)
binary_frame_dilating_iterations = 2
kernel_hand_made = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                           ], np.uint8)
use_kernel_hand_made = 1
hand_made_dilation_iterations = 3

# --- Matting Parameters --- #

r_alpha = -1
background_inner_radius_matting = 50
background_interval_radius_matting = 350

foreground_outer_radius_matting = 60
foreground_interval_radius_matting = 70

outer_radius_uncertain_area_matting = 0
inner_radius_uncertain_area_matting = 50

cost_field_exp_coef = 0.5

# --- Tracking Parameters --- #

N = 40 # Number of particles in the particle filter. Higher value for better tracking but slower performance
particles_noise_variance = 6 # Variance for the particle predicting. If the object moves quickly, set a high value
