import cv2
from CODE.config import *

import os

global ratio
global video_borders_coef
global T_buffer
#global gaussian_anti_aliasing_blur_size

dirname = os.path.dirname(os.path.dirname(__file__))

""" This function gets a frame from a video which is in stabilization process,
    and fixes the borders by enclosing the video.
    """
def fixBorder(frame):

    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, video_borders_coef)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

""" This function gets a curve and a radius and performs moving
    average. Used for smoothing the stabilized video.
"""
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size)/window_size

    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed

""" This function gets a trajectory between frames and
    returns a smoothed trajectory.
    """
def smooth(trajectory):
    smoothed_traj = np.copy(trajectory)
    for i in range(3):
        smoothed_traj[:, i] = movingAverage(trajectory[:, i], radius=smoothness_coef)

    return smoothed_traj

""" This function gets an input video, and parameters for the video stabilization
    and uses cv2.goodFeaturesToTrack, cv2.calcOpticalFlowPyrLK and  cv2.estimateAffinePartial2D
    to estimate the transformations between the frames."""
def EstimateTransforms(input_video, maxCorners, qualityLevel, minDistance, starting_frame):
    #global gaussian_anti_aliasing_blur_coef
    input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    num_of_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(round(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # Pre-define transformation-store array
    transforms = np.zeros((num_of_frames - 1, 3), np.float32)
    for j in range(starting_frame):
        success, prev_frame = input_video.read()
        if ratio != 1:
            prev_frame = cv2.GaussianBlur(prev_frame, gaussian_anti_aliasing_blur_size, gaussian_anti_aliasing_blur_coef)
            prev_frame = cv2.resize(prev_frame, dsize=(int(round(width / ratio)), int(round(height / ratio))))
        print(j)
    if starting_frame == 0:
        success, prev_frame = input_video.read()
        if ratio != 1:
            prev_frame = cv2.GaussianBlur(prev_frame, gaussian_anti_aliasing_blur_size, gaussian_anti_aliasing_blur_coef)
            prev_frame = cv2.resize(prev_frame, dsize=(int(round(width / ratio)), int(round(height / ratio))))
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if not success:
        return
    else:

        previous_precentage = 0
        tkinter_display("Stabilization - EstimateTransforms - {0}% done".format(previous_precentage))
        for i in range(starting_frame, num_of_frames - 1):
            precentage = round(100*(i+1)/(num_of_frames-1))
            if previous_precentage != precentage:
                tkinter_display("Stabilization - EstimateTransforms - {0}% done".format(precentage))

            previous_precentage = precentage

            # Detect feature points in previous frame

            prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners, qualityLevel, minDistance, blockSize=5)

            # Read next frame
            success, curr_frame = input_video.read()
            if not success:
                break

            if ratio != 1:
                curr_frame = cv2.GaussianBlur(curr_frame, gaussian_anti_aliasing_blur_size, gaussian_anti_aliasing_blur_coef)
                curr_frame = cv2.resize(curr_frame, dsize=(int(round(width / ratio)), int(round(height / ratio))))
            # Convert to grayscale
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow (i.e. track feature points)
            curr_frame_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_frame_points, None)



            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_frame_points = prev_frame_points[idx]
            curr_frame_points = curr_frame_points[idx]

            # Find transformation matrix
            trans = cv2.estimateAffinePartial2D(prev_frame_points, curr_frame_points)[0]

            # Extract traslation
            dx = trans[0, 2]
            dy = trans[1, 2]

            # Extract rotation angle
            da = np.arctan2(trans[1, 0], trans[0, 0])

            # Store transformation
            transforms[i] = [dx, dy, da]

            # Move to next frame
            prev_frame_gray = curr_frame_gray

        return transforms

""" This function gets an input video and parameters, and creates
    a stabilized video. This is the stabilization's main function that is being
    called from the GUI. """
def StabilizeVideo(input_video, maxCorners, qualityLevel, minDistance):
    starting_frame = 0
    num_of_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = int(round(width/ratio))
    height = int(round(height / ratio))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_vid = cv2.VideoWriter(os.path.join(dirname, 'Output/stabilize.avi'), fourcc, 30.0, (width, height))

    transforms = EstimateTransforms(input_video, maxCorners, qualityLevel, minDistance, starting_frame)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_traj = smooth(trajectory)
    diff = smoothed_traj - trajectory
    transforms_smooth = transforms + diff

    input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for j in range(starting_frame):
        success, frame = input_video.read()
        if ratio!=1:
            frame = cv2.GaussianBlur(frame, gaussian_anti_aliasing_blur_size, gaussian_anti_aliasing_blur_coef)
            frame = cv2.resize(frame, dsize=(width,height))
        frame = fixBorder(frame)
        output_vid.write(frame)
    for i in range(starting_frame, num_of_frames - 2):
        # Read next frame
        success, frame = input_video.read()

        if ratio!=1:
            frame = cv2.GaussianBlur(frame, gaussian_anti_aliasing_blur_size, gaussian_anti_aliasing_blur_coef)
            frame = cv2.resize(frame, dsize=(width,height))
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        trans = np.zeros((2, 3), np.float32)
        trans[0, 0] = np.cos(da)
        trans[0, 1] = -np.sin(da)
        trans[1, 0] = np.sin(da)
        trans[1, 1] = np.cos(da)
        trans[0, 2] = dx
        trans[1, 2] = dy

        # Apply affine wrapping to the given frame
        stabilized_frame = cv2.warpAffine(frame, trans, (width, height))

        # Fix border artifacts
        stabilized_frame = fixBorder(stabilized_frame)

        output_vid.write(stabilized_frame)
    output_vid.release()
    return output_vid
