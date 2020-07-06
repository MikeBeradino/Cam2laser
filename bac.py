#!/usr/bin/env python
import pyrealsense2 as rs
import numpy as np
import cv2
from asciisciit import AsciiImage
import os
import random
import time

cascPath = "./src/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def converter():

    ################################################################
    # used for laser val
    ###############################################################
    old_value_lase = 0
    old_min_lase = 0
    old_max_lase = 255
    new_min_lase = 500
    new_max_lase = 4095

    ################################################################
    # used for maping xy
    ###############################################################
    old_value = 0
    old_min = 0
    old_max = 160
    new_min = 0
    new_max = 4095

    ################################################################
    string_array=[]
    ################################################################
    #folder imfo
    ################################
    img_folder_path = 'maked_dump/'
    dirListing = os.listdir(img_folder_path)
    filenum = len(dirListing)
    intfilenum = int(filenum)
    #print(intfilenum)
    #############################
    x_ran = 2700
    for i in range(0, intfilenum):
        x_ran -= 700
        #x_ran = random.randrange(-2000,2000)
        #y_ran = random.randrange(0,1000)
        

        # Input image

        filename_ = str(i)
        input = cv2.imread("maked_dump/"+ filename_+".jpg")
        # Get input size
        width, height, _ = input.shape

        # Desired "pixelated" size
        w, h = (160,160)

        # Resize input to "pixelated" size
        temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

        # Initialize output image
        #output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        img_rotate_180 = cv2.rotate(temp, cv2.ROTATE_180)

        gray = cv2.cvtColor(img_rotate_180, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("for_laser/converted"+filename_+".jpg" ,gray)

        #img = cv2.imread(for_laser/converted"+filename_+".jpg",0)
        
        # probably want to give one name to file iff using the mirror
        #f= open( "textfiles/"+str (filename_)+ ".txt","w+")
        f= open( "lasershark_hostapp/txt_frames/"+str (filename_)+ ".txt","w+")
        
        #remove the line below to  keep all files in array
        string_array=[]
        
        rows,cols = gray.shape
        line = ("r=1000" +"\n"+"e=1"+"\n")
        f.write(line)
        for l in range(rows):

            
            for j in range(cols):
                max_color =0
                color = gray[l,j]
                

                xpos = str(j)
                ypos = str(l)
                string_color = str(color)
                ################################################################
                # used for maping xy
                ################################################################
                if (string_color != "255" and color > 50):
                    if (old_min != old_max) and (new_min != new_max):
                        new_value_x= (((j- old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
                        new_value_y= (((l - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
                ################################################################
                # used for maping laser
                ################################################################
                    if (old_min_lase != old_max_lase) and (new_min_lase != new_max_lase):
                        new_value_x_lase= (((color- old_min_lase) * (new_max - new_min_lase)) / (old_max_lase - old_min_lase)) + new_min_lase
                    
                    x_boundry = new_value_x + x_ran
                    y_boundry = new_value_y 
                    if (x_boundry < 4095 and x_boundry >0  ):
                        #new_value_y_ran = new_value_y + y_ran
                        #new_value_x_ran = new_value_x + x_ran
                        new_value_x_str = str(x_boundry)
                        new_value_y_str = str(y_boundry)
                        new_value_x_lase_str =str(new_value_x_lase)


                        line2 = ("s="+ new_value_x_str+","+new_value_y_str+",0,0,0,0" +"\n"+"s="+ new_value_x_str+","+new_value_y_str+","+new_value_x_lase_str+",0,0,0" +"\n"+ "s="+ new_value_x_str+","+new_value_y_str+",0,0,0,0" +"\n")
                        string_array.append (line2)        
        
        random.shuffle(string_array)
        for lines in string_array:
            f.write((lines)) # works with any number of elements in a line
        
        line = ("f=1" +"\n"+"e=0")
        #mirro move
        
        f.write(line)
        f.close() 


def capture():
    framecap = 0
    framecap_count =0
    start_capture = False


    #img = AsciiImage("lena.jpg", scalefactor=0.2)
    #print(img)
    #img.render("output.png", fontsize=8, bg_color=(20,20,20), fg_color=(255,255,255))

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.2 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            hole_filling = rs.hole_filling_filter()
            filled_depth = hole_filling.process(aligned_depth_frame)
            

            # Validate that both frames are valid
            if not filled_depth or not color_frame:
                continue

            depth_image = np.asanyarray(filled_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())




            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 255
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
           
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            #need to fix this
            bg_removed2 = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 0, color_image)
            # Render images for color       
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_HSV)


            # crop the image using array slices -- it's a NumPy array
            # after all!
            cropped = bg_removed[0:720, 280:1000]
            cropped2 = bg_removed2[0:720, 280:1000]


            ################################################################
            # open cv stuff
            ################################################################

            #color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )   
        
            far=""
            farColor=(0,0,0)
            

            ################################################################
            # open cv stuff
            ################################################################

            ##############################################################################
            # ASCII on term 
            ##############################################################################
            img = AsciiImage(cropped2, scalefactor=0.19,)
            print(img)   
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                dist =  aligned_depth_frame.get_distance(x, y)
                if  (dist<clipping_distance_in_meters):
                    #far = "range" #far
                    #farColor=(0,255,0)
                    start_capture = True

                #cv2.putText(color_image,far,(5,15),5, 1,farColor, lineType=cv2.LINE_AA)
                #cv2.rectangle(color_image, (x, y), (x+w, y+h),farColor, 2)
            
            ##############################################################################
            # saving no background images
            ##############################################################################
            if framecap < 7 and start_capture == True:
                framecap_count = framecap_count +1
                if framecap_count > 2:
                    framecap_string =  str (framecap)
                    cv2.imwrite("maked_dump/"+ framecap_string+".jpg" ,cropped)
                    framecap = framecap + 1 
                    framecap_count = 0
             ##############################################################################      


            images = np.hstack((bg_removed, depth_colormap))
           
            # create windows 
            #cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('Align Example', images)
            #cv2.imshow('RealSense', color_image)
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if framecap == 7 or key & 0xFF == ord('q') or key == 27:
                #converter()
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

def lase():
    ################################################################
    #folder imfo
    ################################
    img_folder_path = 'lasershark_hostapp/txt_frames/'
    dirListing = os.listdir(img_folder_path)
    filenum = len(dirListing)
    intfilenum = int(filenum)
    #print(intfilenum)
    #############################

    for x in range (0,intfilenum):
        #print (x)
        time.sleep( 20 )
        string_x = str(x)
        os.system("cat '------put your dir path here-------"+string_x+".txt'  | ./lasershark_hostapp/lasershark_stdin")
        
#converter()

while True:
    capture()
    converter()
    lase()
