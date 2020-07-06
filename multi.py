#!/usr/bin/python
import time
import os

################################################################
#folder imfo
################################
img_folder_path = 'lasershark_hostapp/txt_frames/'
dirListing = os.listdir(img_folder_path)
filenum = len(dirListing)
intfilenum = int(filenum)
print(intfilenum)
#############################

for x in range (6,intfilenum):
	print (x)
	time.sleep( 1 )
	string_x = str(x)
	#os.system("cat '/home/artshow/Desktop/realsence_c++/lasershark_hostapp/txt_frames/"+string_x+".txt'  | ./lasershark_hostapp/lasershark_stdin")
	os.system("cat '/home/artshow/Desktop/realsence_c++/lasershark_hostapp/txt_frames/2.txt'  | ./lasershark_hostapp/lasershark_stdin")