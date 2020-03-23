
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pdb
import os
import random

detector = dlib.get_frontal_face_detector()
predictor = "shape_predictor_68_face_landmarks.dat"
dir_images = "images/input/"
for this_image in os.listdir(dir_images):
	# this_image = random.choice()
	# print this_image
	my_image = dir_images+this_image
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(my_image)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# print shape
		# pdb.set_trace()
		

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		# loop over the (x, y)-coordinates for the jaw line
		# and connect them on the image
		for h in range(1,17):
			start_point = (shape[h-1][0],shape[h-1][1])
			end_point = (shape[h ][0],shape[h ][1])
			# Green color in BGR 
			color = (0, 255, 0) 
			# Line thickness of 9 px 
			thickness = 1
			image = cv2.line(image, start_point, end_point, color, thickness) 
	cv2.imwrite("images/output/"+this_image,image)
