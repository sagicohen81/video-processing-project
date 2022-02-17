import cv2
import matplotlib.pyplot as plt
import numpy.matlib
import matplotlib.patches as patches
from CODE.MouseImageInput import *
from CODE.config import *
from tkinter import messagebox
import os

dirname = os.path.dirname(os.path.dirname(__file__))

""" INPUT  = S_next_tag (previously sampled particles)
    OUTPUT = S_next (predicted particles. weights and CDF not updated yet)
"""
def predictParticles(S_next_tag, num_of_fields, N, height, width):
    global particles_noise_variance
    S_next = S_next_tag.copy()
    temp = np.shape(S_next_tag)
    if(np.shape(S_next_tag) == (1,num_of_fields*N)):
        s_init = S_next_tag.T[0:6].T
        S_next = np.matlib.repmat(s_init, N, 1).T

    # Apply Dynamic Model
    S_next[0][:] += S_next[4][:]
    S_next[1][:] += S_next[5][:]

    # Add White Noise
    for n in range(num_of_fields):
        for m in range(N):
            if n != 2 and n != 3:
                S_next[n][m] += int(np.round(np.random.normal(0, particles_noise_variance)))

    # Check For Overflow
    for i in range(N):
        if S_next[0][i] < S_next[2][0]:
            S_next[0][i] = S_next[2][0]
        if S_next[1][i] < S_next[3][0]:
            S_next[1][i] = S_next[3][0]
        if S_next[0][i] > width - S_next[2][0]:
            S_next[0][i] = width - S_next[2][0] - 1
        if S_next[1][i] > height - S_next[3][0]:
            S_next[1][i] = height - S_next[3][0] - 1
    return S_next


""" INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
    OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1
    VECTOR. NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)
"""
def compNormHist(I, s_initial):
    quant_coef = 16
    hist_3d_size = 16
    hist_size = 4096
    I_copy = I.copy()
    hist_3d = np.zeros((hist_3d_size,hist_3d_size,hist_3d_size), dtype = int)

    I_r = I_copy[:,:,0]
    I_g = I_copy[:,:,1]
    I_b = I_copy[:,:,2]
    left_up = [s_initial[1] - s_initial[3], s_initial[0] - s_initial[2]]
    right_up = [s_initial[1] + s_initial[3], s_initial[0] - s_initial[2]]
    left_down = [s_initial[1] - s_initial[3], s_initial[0] + s_initial[2]]
    right_Down = [s_initial[1] + s_initial[3], s_initial[0] + s_initial[2]]

    I_r_window = I_r[np.ix_(range(int(left_up[0]),int(right_up[0])), range(int(left_up[1]), int(left_down[1])))]
    I_g_window = I_g[np.ix_(range(int(left_up[0]),int(right_up[0])), range(int(left_up[1]), int(left_down[1])))]
    I_b_window = I_b[np.ix_(range(int(left_up[0]),int(right_up[0])), range(int(left_up[1]), int(left_down[1])))]
    #print("{0},{1},{2}".format(left_up,right_up,left_down))
    for i in range(int(s_initial[3]*2)):
        for j in range(int(s_initial[2]*2)):
            I_r_window[i, j] = I_r_window[i, j] / quant_coef
            I_g_window[i, j] = I_g_window[i, j] / quant_coef
            I_b_window[i, j] = I_b_window[i, j] / quant_coef

    for i in range(int(s_initial[3] * 2)):
        for j in range(int(s_initial[2] * 2)):
            hist_3d[I_r_window[i,j], I_g_window[i,j], I_b_window[i,j]] += 1

    hist = np.reshape(hist_3d, (1,4096))

    hist_sum = np.sum(hist)
    norm_hist = hist/hist_sum

    return norm_hist


""" INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
    OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)
"""
def compBatDist(p, q):
    p_size = 4096
    sum = 0
    for i in range(p_size):
        sum += np.sqrt(p.T[i]*q.T[i])

    W = np.exp(20*sum)
    return W


""" INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
    OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX)
"""
def sampleParticles(S_prev, C, num_of_fields, N):
    #N = 100
    #fields = 6
    s_next_tag = np.zeros(np.shape(S_prev))

    for n in range(N):

        r = np.random.uniform(0, 1)
        for j in range(N):
            if(C[j] >= r):
                s_next_tag[:, n] = S_prev[:, j]
                break

    # Check For Overflow
    for i in range(N):
        if(s_next_tag[0, i] < S_prev[2, 0]):
            s_next_tag[0, i] = S_prev[2, 0]
        if (s_next_tag[1, i] < S_prev[3, 0]):
            s_next_tag[1, i] = S_prev[3, 0]
    return s_next_tag


""" This function plots a rectangle of the maximum weight particle on the
    frame and saves it to the output video.
"""
def showPersonTrack(frame, output_vid, S, W, i):
    max_particle_weight = np.amax(W)


    max_particle_index = np.argwhere(W == max_particle_weight)[0]

    max_particle = S[:,max_particle_index]

    max_X_center = max_particle[0]
    max_Y_center = max_particle[1]

    width_div_2 = max_particle[2]
    height_div_2 = max_particle[3]

    left_down_red = (max_X_center - width_div_2, max_Y_center - height_div_2)
    right_up_red = (max_X_center + width_div_2, max_Y_center + height_div_2)

    # Creating Rectangles #

    rect_max_red = patches.Rectangle(left_down_red, width_div_2*2, height_div_2*2, linewidth=1, edgecolor='r', facecolor='none')
    frame_rectangled = cv2.rectangle(frame, left_down_red, right_up_red, [0,255,0])

    output_vid.write(frame_rectangled)


""" This function gets an input video and lets the user to choose an object to follow,
    then uses particle filter to track the object and saves the OUTPUT.avi video.
    This is the main function of the tracking that is being called from the GUI.
"""
def followObjectInVideo(input_video, num_of_fields, N):
    width = int(round(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_of_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = cv2.VideoWriter(os.path.join(dirname, 'Output/OUTPUT.avi'), fourcc, 30.0, (width, height))
    input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # LOAD FIRST IMAGE
    success_cap, first_frame = input_video.read()
    (image_height, image_width, depth) = np.shape(first_frame)

    messagebox.showinfo("Follow Object", "Please select the object to follow.\nPress and hold the left mouse button, then select the rectangle and release.\n Press 'c' when ready or 'r' to reset.")
    x_center, y_center, obj_width, obj_height = getInputFromImageClick(first_frame, 1)
    half_width = int(round(obj_width/2))
    half_height = int(round(obj_height/2))

    tkinter_display("Tracking - followObjectInVideo - 0% done")

    # Initial Settings
    s_initial = [x_center,
                 y_center,
                 half_width,
                 half_height,
                 0,
                 0]

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
    S = predictParticles(np.matlib.repmat(s_initial, 1, N), num_of_fields, N, image_height, image_width)

    # COMPUTE NORMALIZED HISTOGRAM
    q = compNormHist(first_frame, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(N)
    C = np.zeros(N)
    for particle in range(N):
        temp = S[:, particle]
        p = compNormHist(first_frame, S[:, particle])
        W[particle] = compBatDist(p, q)

    W_sum = np.sum(W)
    W = W / W_sum

    C = np.cumsum(W)
    showPersonTrack(first_frame, output_vid, S, W, 0)
    images_processed = 1

    # LOAD NEW IMAGE FRAME
    success_cap, curr_frame = input_video.read()

    previous_precentage = 0

    # MAIN TRACKING LOOP
    while success_cap == True:
        precentage = round(100*(images_processed/(num_of_frames-1)))

        if precentage!= previous_precentage:
            tkinter_display("Tracking - followObjectInVideo - {0}% done".format(precentage))
        previous_precentage = precentage

        S_prev = S

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sampleParticles(S_prev, C, num_of_fields, N)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE)
        S = predictParticles(S_next_tag, num_of_fields, N, image_height, image_width)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        W = np.zeros(N)
        C = np.zeros(N)
        for particle in range(N):
            temp = S[:, particle]
            p = compNormHist(curr_frame, S[:, particle])
            W[particle] = compBatDist(p, q)

        W_sum = np.sum(W)
        W = W / W_sum

        C = np.cumsum(W)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        i = images_processed
        if 0 == images_processed % 1:
            showPersonTrack(curr_frame, output_vid, S, W, i)

        success_cap, curr_frame = input_video.read()

    output_vid.release()
