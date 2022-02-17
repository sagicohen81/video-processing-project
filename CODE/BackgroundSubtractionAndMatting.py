import cv2
from scipy.stats import gaussian_kde
import scipy.signal
from CODE.MouseImageInput import *
from CODE.ParticleFilterTracking import *
from CODE.WDT import *
from CODE.config import *

import os

dirname = os.path.dirname(os.path.dirname(__file__))

""" This function gets an input video and a kernel size,
    and returns a list of frames which are the median of
    each frame. """
def findMedianVideo(input_video, kernel_size):
    num_of_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = cv2.VideoWriter('MedianVideo.avi', fourcc, 30.0, (width, height))

    frames_list = np.zeros((num_of_frames + 2*kernel_size, height, width, 3))


    input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- fill matrix with video frames --- #
    frame_num = 0
    pad_frame_num = 0
    curr_success, frame_read = input_video.read()
    while curr_success:
        frame_read = cv2.resize(frame_read, dsize=(width, height))
        frames_list[frame_num + kernel_size,:,:,:] = frame_read
        if frame_num > num_of_frames - kernel_size:
            frames_list[pad_frame_num] = frame_read
            pad_frame_num += 1
        curr_success, frame_read = input_video.read()
        frame_num += 1

    input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # --- pad after the end --- #
    frame_num = 0
    curr_success, frame_read = input_video.read()
    while frame_num < kernel_size:
        frame_read = cv2.resize(frame_read, dsize=(width, height))
        frames_list[num_of_frames + kernel_size + frame_num, :, :, :] = frame_read
        curr_success, frame_read = input_video.read()
        frame_num += 1
    #input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    med_frames_list = np.zeros(frames_list.shape)
    previous_precentage = 0
    for i in range(height):
        for j in range(width):
            precentage = round(100* ((i+1)/height))
            if precentage != previous_precentage:
                tkinter_display("Background Subtraction - findMedianVideo - {0}% done".format(precentage))
                previous_precentage = precentage
            for color in range(3):
                #a = frames_list[0][i,j][color]
                med_frames_list[:, i, j, color] = scipy.signal.medfilt(np.array(frames_list[:, i, j, color]), kernel_size)
    """
    frame_num = 0
    for med_frame in med_frames_list:
        if frame_num > kernel_size and frame_num < kernel_size + num_of_frames + 1:
            med_frame_copy = np.uint8(med_frame)
            output_vid.write(med_frame_copy)
        frame_num += 1
    """
    return med_frames_list[kernel_size: kernel_size + num_of_frames + 1, :, :, :]


""" This function uses the gaussian_kde scipy's function
    to return an estimated density estimation on a supplied
    grid. """
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


""" This function gets scribble points both for the
    foreground and the background, and returns their
    density estimation on the 0-256 grid. """
def getKDEForeAndBack(scribble_pts_fore, scribble_pts_back):
    colors_grid = np.linspace(0, 255, 256)
    kde_fore = kde_scipy(scribble_pts_fore, colors_grid)
    kde_back = kde_scipy(scribble_pts_back, colors_grid)

    return kde_fore, kde_back


""" This function gets a frame and its binary frame (indicates where the foreground is),
    and returns scribble points for the foreground and background on areas defined
    by user."""
def findArtifficialScribblesForMatting(frame_v, binary_frame):

    # --- calculate scribbles area for the background --- #
    binary_frame = np.uint8(binary_frame)
    kernel = np.ones((background_inner_radius_matting, background_inner_radius_matting), np.uint8)
    mat_erode = cv2.erode(np.uint8(binary_frame), kernel, iterations=1)

    #kernel = np.ones((dilation_inner_radius_matting + dilation_interval_radius_matting, dilation_inner_radius_matting + dilation_interval_radius_matting), np.uint8)
    kernel = np.ones((background_interval_radius_matting, background_interval_radius_matting), np.uint8)
    mat_dilated = cv2.dilate(mat_erode, kernel, iterations=1)

    mask_background_scribbles = np.where(
        np.logical_and(np.logical_or(mat_dilated == 1, mat_erode == 0), binary_frame == 0), 1, 0)

    mask_background_scribbles = np.where(np.logical_or(np.logical_and(mat_erode == 0, binary_frame == 1), np.logical_and(mat_dilated == 1, binary_frame == 0)), 1, 0)

    # --- calculate scribbles area for the foreground --- #
    kernel = np.ones((foreground_outer_radius_matting, foreground_outer_radius_matting), np.uint8)
    mat_eroded_first = cv2.erode(np.uint8(binary_frame), kernel, iterations=1)

    kernel = np.ones((foreground_interval_radius_matting, foreground_interval_radius_matting), np.uint8)
    mat_eroded_second = cv2.erode(mat_eroded_first, kernel, iterations=1)

    mask_foreground_scribbles = np.where(
        np.logical_and(mat_eroded_second == 0, mat_eroded_first == 1, binary_frame == 1), 1, 0)

    #plt.imshow(binary_frame)
    #plt.show()
    #plt.imshow(mask_background_scribbles)
    #plt.show()
    # --- calculate values and indices of the scribble points --- #
    foreground_scribbles_indexes = [(index, list(row).index(1)) for index, row in enumerate(mask_foreground_scribbles) if 1 in row]
    #foreground_scribbles_values = frame_v[foreground_scribbles_indexes]

    background_scribbles_indexes = [(index, list(row).index(1)) for index, row in enumerate(mask_background_scribbles) if 1 in row]
    #background_scribbles_values = frame_v[background_scribbles_indexes[::-1]]

    foreground_scribbles_values = np.zeros(len(foreground_scribbles_indexes))
    background_scribbles_values = np.zeros(len(background_scribbles_indexes))
    for i in range(len(foreground_scribbles_indexes)):
        foreground_scribbles_values[i] = frame_v[foreground_scribbles_indexes[i]]

    for j in range(len(background_scribbles_indexes)):
        background_scribbles_values[j] = frame_v[background_scribbles_indexes[j]]

    foreground_scribbles_indexes = [t[::-1] for t in foreground_scribbles_indexes]
    background_scribbles_indexes = [t[::-1] for t in background_scribbles_indexes]

    return foreground_scribbles_indexes, background_scribbles_indexes, foreground_scribbles_values, background_scribbles_values


""" This function gets a frame and its binary frame, and calculates its alpha map using 
    scribble points, kde and wdt.
    """
def findAlpha(frame_v, binary_frame):
    # --- find scribble points for estimating alpha, and calculate KDE --- #
    a = np.array_equal(binary_frame, np.zeros(binary_frame.shape))
    b = np.array_equal(binary_frame, np.ones(binary_frame.shape))
    if not np.array_equal(binary_frame, np.zeros(binary_frame.shape)):
        if not np.array_equal(binary_frame, np.ones(binary_frame.shape)):
            foreground_scribbles_indexes, background_scribbles_indexes, foreground_scribbles_values, background_scribbles_values = findArtifficialScribblesForMatting(frame_v, binary_frame)
            kde_alpha_fore, kde_alpha_back = getKDEForeAndBack(foreground_scribbles_values, background_scribbles_values)

            # --- find estimated new PDF for foreground and background --- #
            est_hist_foreground = np.zeros(frame_v.shape) # we'll do it for the whole image for comfort
            est_hist_background = np.zeros(frame_v.shape)
            pdf_on_foreground = np.zeros(frame_v.shape)

            for i in range(np.shape(frame_v)[0]):
                for j in range(np.shape(frame_v)[1]):
                    est_hist_foreground[i, j] = kde_alpha_fore[frame_v[i, j]]
                    est_hist_background[i, j] = kde_alpha_back[frame_v[i, j]]
                    pdf_on_foreground[i, j] = est_hist_foreground[i, j] / (est_hist_foreground[i, j] + est_hist_background[i, j] + 10 ** -7)

            pdf_on_background = np.ones(pdf_on_foreground.shape) - pdf_on_foreground

            # --- calculate WDT --- #
            sobel_x_pdf_fore = cv2.Sobel(pdf_on_foreground, ddepth=cv2.CV_64F, dx=1, dy=0)
            sobel_y_pdf_fore = cv2.Sobel(pdf_on_foreground, ddepth=cv2.CV_64F, dx=0, dy=1)
            sobel_pdf_fore = np.sqrt(sobel_x_pdf_fore ** 2 + sobel_y_pdf_fore ** 2)

            sobel_x_pdf_back = cv2.Sobel(pdf_on_background, ddepth=cv2.CV_64F, dx=1, dy=0)
            sobel_y_pdf_back = cv2.Sobel(pdf_on_background, ddepth=cv2.CV_64F, dx=0, dy=1)
            sobel_pdf_back = np.sqrt(sobel_x_pdf_back ** 2 + sobel_y_pdf_back ** 2)



            cost_field_fore = map_image_to_costs(sobel_pdf_fore, foreground_scribbles_indexes)
            cost_field_back = map_image_to_costs(sobel_pdf_back, background_scribbles_indexes)

            distance_transform_fore = get_weighted_distance_transform(cost_field_fore)
            distance_transform_back = get_weighted_distance_transform(cost_field_back)

            # --- calculate alpha --- #

            omega_fore = np.multiply(np.power((distance_transform_fore + 10**-3), -r_alpha), pdf_on_foreground)
            omega_back = np.multiply(np.power((distance_transform_back+ 10**-3), -r_alpha), pdf_on_background)

            alpha = omega_fore / (omega_fore+omega_back + 10**-3)

        else:
            alpha = np.ones(binary_frame.shape)
    else:
        alpha = np.zeros(binary_frame.shape)

    return alpha


""" This function gets a frame, its binary frame, a new background and an alpha
    map, and returns an image containing the foreground with the new background,
    calculated according to alpha map"""
def getMattedImage(frame_bgr, new_background, binary_frame, alpha):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    binary_frame_3d = np.zeros((np.shape(binary_frame)[0], np.shape(binary_frame)[1], 3))
    binary_frame_3d[:, :, 0] = binary_frame
    binary_frame_3d[:, :, 1] = binary_frame
    binary_frame_3d[:, :, 2] = binary_frame

    binary_frame = np.uint8(binary_frame)

    kernel = np.ones((outer_radius_uncertain_area_matting, outer_radius_uncertain_area_matting), np.uint8)
    binary_eroded_first = cv2.erode(np.uint8(binary_frame), kernel, iterations=1)
    #unidentified_area_dilation = np.where(np.logical_and(binary_frame == 0, binary_dilated == 1), 1, 0)

    kernel = np.ones((inner_radius_uncertain_area_matting, inner_radius_uncertain_area_matting), np.uint8)
    binary_eroded_second = cv2.erode(np.uint8(binary_eroded_first), kernel, iterations=1)
    #unidentified_area_eroding = np.where(np.logical_and(binary_frame == 1, binary_eroded == 0), 1, 0)

    unidentified_area = np.where(np.logical_and(binary_eroded_first == 1, binary_eroded_second == 0, binary_frame == 1), 1, 0)
    foreground = np.where(binary_frame_3d == [1, 1, 1], frame_bgr, 0)

    matted = np.where(binary_frame_3d == [1, 1, 1], foreground, new_background)


    matted[:, :, 0] = np.where(unidentified_area == 1, np.round(alpha * frame_bgr[:, :, 0] + (1 - alpha) * new_background[:, :, 0]).astype(np.uint8), matted[:, :, 0])
    matted[:, :, 1] = np.where(unidentified_area == 1, np.round(alpha * frame_bgr[:, :, 1] + (1 - alpha) * new_background[:, :, 1]).astype(np.uint8), matted[:, :, 1])
    matted[:, :, 2] = np.where(unidentified_area == 1, np.round(alpha * frame_bgr[:, :, 2] + (1 - alpha) * new_background[:, :, 2]).astype(np.uint8), matted[:, :, 2])

    return matted


""" This function gets the stabilized video and creates its binary and extracted
    foreground videos. This is the background subtraction's main function that is being
    called from the GUI.
    """
def createBinaryVideo(stabilized_video):
    width = int(round(stabilized_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(stabilized_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_of_frames = int(stabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))

    med_frames_list = findMedianVideo(stabilized_video, median_video_kernel)

    stabilized_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    median_video = cv2.VideoCapture('MedianVideo.avi')
    #num2 = int(median_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
    binary_vid = cv2.VideoWriter(os.path.join(dirname, 'Output/binary.avi'), fourcc1, 30.0, (width, height))
    extracted_vid = cv2.VideoWriter(os.path.join(dirname, 'Output/extracted.avi'), fourcc2, 30.0, (width, height))

    frame_num = 0

    median_frame_bgr = med_frames_list[frame_num, :, :, :]
    #success1, median_frame_bgr = median_video.read()
    #num_of_frames_median = med_frames_list.shape
    #tkinter_display("{0}".format(num_of_frames_median))
    #time.sleep(100)

    success, frame_bgr = stabilized_video.read()

    binary_frame_3d = np.zeros((np.shape(median_frame_bgr)[0], np.shape(median_frame_bgr)[1], 3))

    while frame_num < num_of_frames and success:
        frame_num += 1
        frame_bgr = cv2.resize(frame_bgr, dsize=(width, height))
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, frame_v = cv2.split(frame_hsv)

        median_frame_bgr = np.uint8(median_frame_bgr)
        median_frame_hsv = cv2.cvtColor(median_frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, median_frame_v = cv2.split(median_frame_hsv)

        gauss_median_frame_v = cv2.GaussianBlur(median_frame_v, median_frame_gaussian_blur_size, median_frame_gaussian_blur_sigma)

        binary_frame = np.where(abs(frame_v - gauss_median_frame_v) > binary_frame_threshold, 1, 0)

        kernel = np.ones(binary_frame_eroding_size, np.uint8)

        binary_frame = cv2.erode(np.uint8(binary_frame), kernel, iterations=binary_frame_eroding_iterations)

        kernel = np.ones(binary_frame_dilating_size, np.uint8)

        binary_frame = cv2.dilate(np.uint8(binary_frame), kernel, iterations=binary_frame_dilating_iterations)

        if use_kernel_hand_made:
            binary_frame = cv2.dilate(np.uint8(binary_frame), kernel_hand_made, iterations=hand_made_dilation_iterations)

        binary_frame_3d[:, :, 0] = binary_frame * 255
        binary_frame_3d[:, :, 1] = binary_frame * 255
        binary_frame_3d[:, :, 2] = binary_frame * 255

        binary_frame_3d = np.where(binary_frame_3d > [240, 240, 240], 255, 0)

        foreground = np.where(binary_frame_3d == [255, 255, 255], frame_bgr, 0)

        binary_frame_3d = np.uint8(binary_frame_3d)
        foreground = np.uint8(foreground)

        binary_vid.write(binary_frame_3d)
        extracted_vid.write(foreground)

        median_frame_bgr = med_frames_list[frame_num, :, :, :]
        #success1, median_frame_bgr = median_video.read()
        success, frame_bgr = stabilized_video.read()

    binary_vid.release()

    return binary_vid


""" This function gets a stabilized video, its binary video and a new background, and creates
    a video with the original video's foregound matted with the
    new background. This is the matting's main function that is
    being called from the GUI.
    """
def createMattedVideo(stabilized_video, binary_video, new_background):
    width = int(round(stabilized_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(stabilized_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_of_frames_stab = int(stabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))
    num_of_frames_bin = int(binary_video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    matted_vid = cv2.VideoWriter(os.path.join(dirname, 'Output/matted.avi'), fourcc, 30.0, (width, height))

    # --- read matching input frames --- #
    success_stabilized, stabilized_frame = stabilized_video.read()
    success_binary, binary_frame = binary_video.read()
    num_frame = 0
    precentage = 0
    previous_presantage = 0
    tkinter_display("Matting - createMattedVideo - {0}% done".format(precentage))
    while success_stabilized and success_binary:
        precentage = round(100* num_frame/(num_of_frames_stab-1))
        if precentage!= previous_presantage:
            tkinter_display("Matting - createMattedVideo - {0}% done".format(precentage))
        num_frame += 1

        #stabilized_frame = cv2.resize(stabilized_frame, dsize=(width, height))
        binary_frame_1D = binary_frame[:, :, 0]
        binary_frame_1D = np.round(binary_frame_1D/255)

        # --- calculate alpha for the frame --- #
        stabilized_frame_hsv = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2HSV)
        h, s, stabilized_frame_v = cv2.split(stabilized_frame_hsv)
        alpha = findAlpha(stabilized_frame_v, binary_frame_1D)
        #plt.imshow(alpha)
        #plt.show()
        # --- calculate matted frame  --- #
        matted_frame = getMattedImage(stabilized_frame, new_background, binary_frame_1D, alpha)
        #plt.imshow(matted_frame)
        #plt.show()
        matted_vid.write(matted_frame)

        # --- read next matching input frames --- #
        success_stabilized, stabilized_frame = stabilized_video.read()
        success_binary, binary_frame = binary_video.read()
    matted_vid.release()
    return matted_vid

"""
binary_video = cv2.VideoCapture(os.path.join(dirname, 'Output/binary.avi'))
stabilized_video = cv2.VideoCapture(os.path.join(dirname, 'Output/stabilize.avi'))
new_background = cv2.imread(os.path.join(dirname, 'Input/background.jpg'))

width_bin = int(round(binary_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
height_bin = int(binary_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_background = cv2.resize(new_background, dsize=(width_bin, height_bin))
createMattedVideo(stabilized_video, binary_video, new_background)

binary_video.release()
stabilized_video.release()
"""