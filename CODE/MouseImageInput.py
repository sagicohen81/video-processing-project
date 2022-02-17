import cv2

scribble_points = []
scribble_points2 = []
first_point = 1
previous_points = (0, 0)
image = []
ratio_comp = 1
corner1 = (-1, -1)
corner2 = (-1, -1)
clean_image = []
mouse_pressed_flag = 0

""" This function is not used """
def plotPoint(image , point):
    cv2.circle(image, point, 1, (0, 255, 0), -1)
    return image


""" This function gets the rectangle's parameters (x and y of the vertices)
    and plots the rectangle on the screen.
"""
def chooseSqureToFollow(event, x, y, flags, param):
    # grab references to the global variables
    global corner1
    global corner2
    global ratio_comp
    global image
    global clean_image

    if event == cv2.EVENT_LBUTTONDOWN and corner1 == (-1, -1):
        x_real = int(round(ratio_comp * x))
        y_real = int(round(ratio_comp * y))
        corner1 = (x_real, y_real)
        image = plotPoint(image, (x, y))
        clean_image = image.copy()

    elif event == cv2.EVENT_LBUTTONUP and corner2 == (-1, -1):
        x_real = int(round(ratio_comp * x))
        y_real = int(round(ratio_comp * y))
        corner2 = (x_real, y_real)
        image = plotPoint(image, (x, y))

    elif event == cv2.EVENT_MOUSEMOVE and corner1 != (-1, -1) and corner2 == (-1, -1):
        image = clean_image.copy()
        corner1_display = (-1, -1)
        corner1_display0 = int(round(corner1[0]/ratio_comp))
        corner1_display1 = int(round(corner1[1]/ratio_comp))
        cv2.line(image, (corner1_display0, corner1_display1), (x, corner1_display1), (0, 255, 0))
        cv2.line(image, (x, corner1_display1), (x, y), (0, 255, 0))
        cv2.line(image, (x, y), (corner1_display0, y), (0, 255, 0))
        cv2.line(image, (corner1_display0, y), (corner1_display0, corner1_display1), (0, 255, 0))


""" This function is not used """
def drawAndChooseLine(event, x, y, flags, param):
    # grab references to the global variables
    global scribble_points
    global first_point
    global previous_points
    global image
    global mouse_pressed_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed_flag = 1
        if first_point == 1:
            first_point = 0
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            scribble_points = [(x_real, y_real)]
            image = plotPoint(image, (x, y))

        else:
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            if (x_real, y_real) not in scribble_points:
                scribble_points.append((x_real, y_real))
                image = plotPoint(image, (x, y))
            #cv2.line(image, (int(round(previous_points[0] / ratio_comp)), int(round(previous_points[1] / ratio_comp))),(x, y), (0, 255, 0))


    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed_flag = 0

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed_flag == 1:
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            if (x_real, y_real) not in scribble_points:
                scribble_points.append((x_real, y_real))
                image = plotPoint(image, (x, y))


    cv2.imshow("image", image)


""" This function is not used """
def drawAndChooseLine2(event, x, y, flags, param):
    # grab references to the global variables
    global scribble_points2
    global first_point
    global previous_points
    global image
    global mouse_pressed_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed_flag = 1
        if first_point == 1:
            first_point = 0
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            scribble_points2 = [(x_real, y_real)]
            image = plotPoint(image, (x, y))

        else:
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            if (x_real, y_real) not in scribble_points2:
                scribble_points2.append((x_real, y_real))
                image = plotPoint(image, (x, y))
            #cv2.line(image, (int(round(previous_points[0] / ratio_comp)), int(round(previous_points[1] / ratio_comp))),(x, y), (0, 255, 0))


    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed_flag = 0

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed_flag == 1:
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            if (x_real, y_real) not in scribble_points2:
                scribble_points2.append((x_real, y_real))
                image = plotPoint(image, (x, y))
    cv2.imshow("image", image)


""" This function is not used """
def choosePointClick(event, x, y, flags, param): # NOT USED
    # grab references to the global variables
    global scribble_points
    global first_point
    global previous_points
    global image

    if event == cv2.EVENT_LBUTTONDOWN:
        if first_point == 1:
            first_point = 0
            x_real = int(round(ratio_comp*x))
            y_real = int(round(ratio_comp*y))
            scribble_points = [(x_real, y_real)]
            image = plotPoint(image, (x, y))

        else:
            x_real = int(round(ratio_comp * x))
            y_real = int(round(ratio_comp * y))
            previous_points = scribble_points[-1]
            scribble_points.append((x_real, y_real))
            image = plotPoint(image, (x, y))
            cv2.line(image,(int(round(previous_points[0]/ratio_comp)), int(round(previous_points[1]/ratio_comp))), (x, y), (0, 255, 0))
        cv2.imshow("image", image)


    # check to see if the left mouse button was released
    #cv2.imshow("image", image)


""" This function gets a frame, and allows the user to choose
    a square to follow. It also has other modes that are not used
    in the final version of the program. Only mode "1" is relevant.
    """
def getInputFromImageClick(input_image, mode):
    global image
    global scribble_points
    global scribble_points2
    global first_point
    global corner1, corner2
    global ratio_comp
    image = input_image
    #image = cv2.resize(image, (1024, 576))
    image = cv2.resize(image, (int(round(image.shape[1]/ratio_comp)), int(round(image.shape[0]/ratio_comp))))
    clone = image.copy()

    while True:
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("image", image)
        if mode == 0:
            cv2.setMouseCallback("image", choosePointClick)
        elif mode == 1:
            cv2.setMouseCallback("image", chooseSqureToFollow)
        elif mode == 2:
            cv2.setMouseCallback("image", drawAndChooseLine)
        elif mode == 3:
            cv2.setMouseCallback("image", drawAndChooseLine2)

        if key == ord("r"):
            if mode == 0 or mode == 2:
                image = clone.copy()
                scribble_points = []
                first_point = 1
            elif mode == 1:
                image = clone.copy()
                corner1 = (-1, -1)
                corner2 = (-1, -1)
            elif mode == 3:
                image = clone.copy()
                scribble_points2 = []
                first_point = 1

        elif key == ord("c"):
            if (mode == 0 or mode == 2) and len(scribble_points) < 2:
                image = clone.copy()
                scribble_points = []
                first_point = 1
            elif mode == 1 and (corner1 == (-1, -1) or corner2 == (-1, -1)):
                image = clone.copy()
                corner1 = (-1, -1)
                corner2 = (-1, -1)
            elif mode == 3 and len(scribble_points) < 2:
                image = clone.copy()
                scribble_points2 = []
                first_point = 1
            else:
                break

    # close all open windows
    cv2.destroyAllWindows()
    if mode == 0 or mode == 2:
        num_of_points = len(scribble_points)
        return_scribbles = scribble_points.copy()
        return num_of_points, return_scribbles
    elif mode == 1:
        center_x = int(round((corner1[0] + corner2[0])/2))
        center_y = int(round((corner1[1] + corner2[1])/2))
        width = abs(corner1[0] - corner2[0])
        height = abs(corner1[1] - corner2[1])
        return center_x, center_y, width, height
    elif mode == 3:
        num_of_points = len(scribble_points2)
        return num_of_points, scribble_points2.copy()
