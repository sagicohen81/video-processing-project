### Written by Nadav Zilberman & Sagi Cohen      ###
### Tel-Aviv Univeristy - Electrical Engineering ###
### Course : Video Processing                   ###
### Final Project 2019 - Video Matting GUI          ###
# imports #
from tkinter import *       #GUI package
from CODE.Stabilization import *
from CODE.BackgroundSubtractionAndMatting import *
from CODE.ParticleFilterTracking import *
from CODE.config import *
import os

global status_precentage
# defines #
STABLE_OPERATION = 2
BACKSUB_OPERATION = 3
MATT_OPERATION = 4
TRACK_OPERATION = 5


dirname = os.path.dirname(os.path.dirname(__file__))

# initializing parameters #
operation_flag = 0

# the following functions show/hide needed buttons on event and flag desired operation#
# (bounded to a specific button left mouse click, later on code) #
def button_list_1(event):
    global operation_flag
    label2.pack(side=LEFT)
    button5.pack(side=LEFT)
    button6.pack_forget()
    button7.pack_forget()
    button8.pack_forget()
    operation_flag = STABLE_OPERATION

def button_list_2(event):
    global operation_flag
    label2.pack(side=LEFT)
    button5.pack(side=LEFT)
    button6.pack(side=LEFT)
    button7.pack_forget()
    button8.pack_forget()
    operation_flag = BACKSUB_OPERATION

def button_list_3(event):
    global operation_flag
    label2.pack(side=LEFT)
    button5.pack(side=LEFT)
    button6.pack(side=LEFT)
    button7.pack(side=LEFT)
    button8.pack_forget()
    operation_flag = MATT_OPERATION

def button_list_4(event):
    global operation_flag
    label2.pack(side=LEFT)
    button5.pack(side=LEFT)
    button6.pack(side=LEFT)
    button7.pack(side=LEFT)
    button8.pack(side=LEFT)
    operation_flag = TRACK_OPERATION

# defining the functions that run the operations #
# all these functions work according to the button they came from (via flags) #

def i_have_nothing(event):
    original_video = cv2.VideoCapture(os.path.join(dirname, 'Input/INPUT.avi'))
    StabilizeVideo(original_video, maxCorners, qualityLevel, minDistance)
    original_video.release()

    if operation_flag == STABLE_OPERATION:
        tkinter_display("Done! 'stabilize.avi' saved")

    if operation_flag >= BACKSUB_OPERATION:
        stabilized_video = cv2.VideoCapture(os.path.join(dirname, 'Output/stabilize.avi'))
        createBinaryVideo(stabilized_video)
        stabilized_video.release()

        if operation_flag == BACKSUB_OPERATION:
            tkinter_display("Done! 'binary.avi' and 'extracted.avi' saved")

    if operation_flag >= MATT_OPERATION:
        stabilized_video = cv2.VideoCapture(os.path.join(dirname, 'Output/stabilize.avi'))
        binary_video = cv2.VideoCapture(os.path.join(dirname, 'Output/binary.avi'))
        width_bin = int(round(binary_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height_bin = int(binary_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_background = cv2.imread(os.path.join(dirname, 'Input/background.jpg'))
        new_background = cv2.resize(new_background, dsize=(width_bin, height_bin))

        createMattedVideo(stabilized_video, binary_video, new_background)

        stabilized_video.release()
        binary_video.release()

        if operation_flag == MATT_OPERATION:
            tkinter_display("Done! 'matted.avi' saved")

    if operation_flag == TRACK_OPERATION:
        matted_video = cv2.VideoCapture(os.path.join(dirname, 'Output/matted.avi'))
        followObjectInVideo(matted_video, 6, N)
        matted_video.release()

        tkinter_display("Done! 'OUTPUT.avi' saved")

def i_have_stable(event):
    if operation_flag >= BACKSUB_OPERATION:
        stabilized_video = cv2.VideoCapture(os.path.join(dirname, 'Output/stabilize.avi'))
        createBinaryVideo(stabilized_video)
        stabilized_video.release()

        if operation_flag == BACKSUB_OPERATION:
            tkinter_display("Done! 'binary.avi' and 'extracted.avi' saved")

    if operation_flag >= MATT_OPERATION:
        stabilized_video = cv2.VideoCapture(os.path.join(dirname, 'Output/stabilize.avi'))
        binary_video = cv2.VideoCapture(os.path.join(dirname, 'Output/binary.avi'))
        new_background = cv2.imread(os.path.join(dirname, 'Input/background.jpg'))

        width_bin = int(round(binary_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height_bin = int(binary_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_background = cv2.resize(new_background, dsize=(width_bin, height_bin))
        createMattedVideo(stabilized_video, binary_video, new_background)

        stabilized_video.release()
        binary_video.release()
        if operation_flag == MATT_OPERATION:
            tkinter_display("Done! 'matted.avi' saved")

    if operation_flag == TRACK_OPERATION:
        matted_video = cv2.VideoCapture(os.path.join(dirname, 'Output/matted.avi'))
        followObjectInVideo(matted_video, 6, N)
        matted_video.release()

        tkinter_display("Done! 'OUTPUT.avi' saved")

def i_have_binary(event):
    if operation_flag >= MATT_OPERATION:
        binary_video = cv2.VideoCapture(os.path.join(dirname, 'Output/binary.avi'))
        stabilized_video = cv2.VideoCapture(os.path.join(dirname, 'Output/stabilize.avi'))
        new_background = cv2.imread(os.path.join(dirname, 'Input/background.jpg'))

        width_bin = int(round(binary_video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height_bin = int(binary_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_background = cv2.resize(new_background, dsize=(width_bin, height_bin))
        createMattedVideo(stabilized_video, binary_video, new_background)

        binary_video.release()
        stabilized_video.release()

        if operation_flag == MATT_OPERATION:
            tkinter_display("Done! 'matted.avi' saved")

    if operation_flag == TRACK_OPERATION:
        matted_video = cv2.VideoCapture(os.path.join(dirname, 'Output/matted.avi'))
        followObjectInVideo(matted_video, 6, N)
        matted_video.release()

        tkinter_display("Done! 'OUTPUT.avi' saved")

def i_have_matted(event):
    if operation_flag == TRACK_OPERATION:
        matted_video = cv2.VideoCapture(os.path.join(dirname, 'Output/matted.avi'))
        followObjectInVideo(matted_video, 6, N)
        matted_video.release()

        tkinter_display("Done! 'OUTPUT.avi' saved")

# initialization of the GUI #
root =Tk()
root.title("Project GUI")
w = 600
h = 120
x = 1200
y = 100
root.geometry("%dx%d+%d+%d" % (w, h, x, y)) # width x height + x_offset + y_offset (no spaces!)
root.resizable(False,False)

#labels and frames declaration #
mainLabel = Label(root, text="         Please choose the operation you want to perform, and then the input you have      ")
mainLabel.pack()
emptyLabel1 = Label(root, text="")
emptyLabel1.pack(side=TOP)
topFrame = Frame(root)
topFrame.pack()
botFrame = Frame(root)
botFrame.pack(side=BOTTOM)
label1 = Label(topFrame, text="Choose Operation :   ")
label1.pack(side=LEFT)
label2 = Label(botFrame, text="Please select the input you already have :   ")

# buttons declaration #
button1 = Button(topFrame,text="Stabilize Video")
button2 = Button(topFrame,text="Back-Subtract Video")
button3 = Button(topFrame,text="Matt Video")
button4 = Button(topFrame,text="Track Object")
button5 = Button(botFrame,text="Original Video")
button6 = Button(botFrame,text="Stabilized Video")
button7 = Button(botFrame,text="Binary Video")
button8 = Button(botFrame,text="Matted Video")

# buttons that always appear on screen #
button1.pack(side=LEFT)
button2.pack(side=LEFT)
button3.pack(side=LEFT)
button4.pack(side=LEFT)

# binding the additional buttons functions to those who already appear #
button1.bind('<Button-1>',button_list_1)    # right argument is the function that needs to be done
button2.bind('<Button-1>',button_list_2)    # left argument takes the eveny needed in order to call the function
button3.bind('<Button-1>',button_list_3)    # '<Button-1>' refers to left mouse click
button4.bind('<Button-1>',button_list_4)

# binding the chosen buttons to the operation functions #
button5.bind('<Button-1>',i_have_nothing)
button6.bind('<Button-1>',i_have_stable)
button7.bind('<Button-1>',i_have_binary)
button8.bind('<Button-1>',i_have_matted)


initialize_status_bar(root)

root.mainloop()     # keeping the GUI window open