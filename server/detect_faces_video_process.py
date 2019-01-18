# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from scipy.spatial import distance as dist
from scipy.signal import argrelextrema
from scipy import ndimage

from skimage.measure import compare_ssim as ssim
from imutils.video import VideoStream
from imutils import face_utils
from PIL import Image
from skimage import measure

from multiprocessing import Pool, Queue
from multiprocessing import Process, Manager
from sqlite3 import Error

import progressbar        # show progress
from tqdm import tqdm     # show progress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import jenkspy    # computer natural breaks: https://pypi.python.org/pypi/jenkspy/0.1.0
import imutils
import sqlite3
import dlib
import glob
import time
import cv2

class MyClass:
	# ---------------------------------------------------------------
	#                        histgram Comparison
	# ---------------------------------------------------------------
	def histCompare(prevFrame, curtFrame, binSize, SET_IND, METHOD_IND, modelSelect):
		# method 0: initialize OpenCV methods for histogram comparison
		OPENCV_METHODS = (
		("Correlation", cv2.HISTCMP_CORREL),
		("Chi-Squared", cv2.HISTCMP_CHISQR),
		("Intersection", cv2.HISTCMP_INTERSECT),
		("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

		# method 1: initialize the scipy methods to compaute distances
		SCIPY_METHODS = (
		("Euclidean", dist.euclidean),
		("Manhattan", dist.cityblock),
		("Chebysev", dist.chebyshev))

		# method 2:  chi-squared distance
		# See in later implementation

		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update the index
		# ---------- remove black pixels in hist calculation
		# usage:  cv2.calcHist(images, channels, mask, histSize, ranges)
		blackThreshold = 0
		# binNum = [16,16,16]
		# binNum = [32,32,32]
		# binNum = [64, 64, 64]

		binNum = [binSize, binSize, binSize]
		segCnt = 0
		segCntPlus = 15

		# modelSelect: 0 ---  RGB model
		# mdoelSelect: 1 ---  HSV model
		if modelSelect == 0:
			prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2RGB) # changed Apr052018. cv2_COLOR_BGR2HSV
			prevMask = np.zeros(prevFrame.shape[:2],dtype="uint8")
			prevMaskInd = np.where((prevFrame[:,:,0] > blackThreshold) | (prevFrame[:,:,1] > blackThreshold) | (prevFrame[:,:,2] > blackThreshold))
			prevMask[prevMaskInd] = 1
			prevHist = cv2.calcHist([prevFrame], [0, 1, 2], prevMask, binNum, [0, 256, 0, 256, 0, 256])
			cv2.normalize(prevHist, prevHist)


			curtFrame = cv2.cvtColor(curtFrame, cv2.COLOR_BGR2RGB)
			curtMask = np.zeros(curtFrame.shape[:2],dtype="uint8")
			curtMaskInd = np.where((curtFrame[:,:,0] > blackThreshold) | (curtFrame[:,:,1] > blackThreshold) | (curtFrame[:,:,2] > blackThreshold))
			curtMask[curtMaskInd] = 1
			curtHist = cv2.calcHist([curtFrame], [0, 1, 2], curtMask, binNum, [0, 256, 0, 256, 0, 256])
			cv2.normalize(curtHist, curtHist)

		elif modelSelect == 1:
			prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2HSV) # changed Apr052018. cv2_COLOR_BGR2HSV
			prevMask = np.zeros(prevFrame.shape[:2],dtype="uint8")
			prevMaskInd = np.where((prevFrame[:,:,0] > blackThreshold) | (prevFrame[:,:,1] > blackThreshold) | (prevFrame[:,:,2] > blackThreshold))
			prevMask[prevMaskInd] = 1
			prevHist = cv2.calcHist([prevFrame], [0, 1, 2], prevMask, binNum, [0, 256, 0, 256, 0, 256])
			cv2.normalize(prevHist, prevHist)
			prevHist = prevHist[:,1:25, :]


			curtFrame = cv2.cvtColor(curtFrame, cv2.COLOR_BGR2HSV)
			curtMask = np.zeros(curtFrame.shape[:2],dtype="uint8")
			curtMaskInd = np.where((curtFrame[:,:,0] > blackThreshold) | (curtFrame[:,:,1] > blackThreshold) | (curtFrame[:,:,2] > blackThreshold))
			curtMask[curtMaskInd] = 1
			curtHist = cv2.calcHist([curtFrame], [0, 1, 2], curtMask, binNum, [0, 256, 0, 256, 0, 256])
			cv2.normalize(curtHist, curtHist)
			curtHist = curtHist[:,1:25, :]

		elif modelSelect == 2:
			prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2HSV) # changed Apr052018. cv2_COLOR_BGR2HSV
			prevMask = np.zeros(prevFrame.shape[:2],dtype="uint8")
			prevMaskInd = np.where((prevFrame[:,:,0] > blackThreshold) & (prevFrame[:,:,1] > blackThreshold) & (prevFrame[:,:,2] > blackThreshold))
			prevMask[prevMaskInd] = 1
			prevHist = cv2.calcHist([prevFrame], [0, 1, 2], prevMask, binNum, [0, 256, 0, 256, 0, 256])
			cv2.normalize(prevHist, prevHist)
			prevHist = prevHist[:,:30:, :]


			curtFrame = cv2.cvtColor(curtFrame, cv2.COLOR_BGR2HSV)
			curtMask = np.zeros(curtFrame.shape[:2],dtype="uint8")
			curtMaskInd = np.where((curtFrame[:,:,0] > blackThreshold) & (curtFrame[:,:,1] > blackThreshold) & (curtFrame[:,:,2] > blackThreshold))
			curtMask[curtMaskInd] = 1
			curtHist = cv2.calcHist([curtFrame], [0, 1, 2], curtMask, binNum, [0, 256, 0, 256, 0, 256])
			cv2.normalize(curtHist, curtHist)
			curtHist = curtHist[:,:30, :]


		prevHist = prevHist.flatten()
		curtHist = curtHist.flatten()
		# choose the method &
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		if SET_IND == 0:
			methodName, method = OPENCV_METHODS[METHOD_IND]
			histDist = cv2.compareHist(prevHist, curtHist, method)
		elif SET_IND == 1:
			methodName, method = SCIPY_METHODS[METHOD_IND]
			histDist = method(prevHist, curtHist)
		elif SET_IND == 2:
			eps = 1e-10
			histDist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(prevHist, curtHist)])

		return histDist, prevHist, curtHist


	# -----------------------------------------------------------
	#                    Edge Detection
	# -----------------------------------------------------------
	def auto_canny(image, sigma=0.33):
		# compute the median of the single channel pixel intensities
		v = np.median(image)

		# apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)

		# return the edged image
		return edged

	def senstiveSkin(prevFace, curtFace, sv, sep):
		fstFace = prevFace.copy()
		sndFace = curtFace.copy()

		# Edge detection - auto canny filter
		fstGray = cv2.cvtColor(fstFace, cv2.COLOR_BGR2GRAY)
		fstBlurred = cv2.GaussianBlur(fstGray, (3, 3), 0)
		fstAuto = auto_canny(fstBlurred)

		sndGray = cv2.cvtColor(sndFace, cv2.COLOR_BGR2GRAY)
		sndBlurred = cv2.GaussianBlur(sndGray, (3, 3), 0)
		sndEdge = auto_canny(sndBlurred)


		diffFace = sndFace - fstFace


		# determine the binary threshold (sv: saturation or brightness channe; seq: 0-1)
		(minVal1, maxVal1, minLoc1, maxLoc1) = cv2.minMaxLoc(prevFace[:,:,sv])
		sepSkin1 = sep*(maxVal1-minVal1) + minVal1

		(minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(curtFace[:,:,sv])
		sepSkin2 = sep*(maxVal2-minVal2) + minVal2

		# threshold the image to reveal saturnation or light regions in frame
		thresh1 = cv2.threshold(prevFace[:,:,sv], sepSkin, 255, cv2.THRESH_BINARY)[1]
		thresh2 = cv2.threshold(curtFace[:,:,sv], sepSkin, 255, cv2.THRESH_BINARY)[1]

		# perform a series of erosions and dilations to remove
		# any small blobs of noise from the thresholded image
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=4)

		# perform a connected component analysis on the thresholded
		# image, then initialize a mask to store only the "large"
		# components
		labels = measure.label(thresh, neighbors=8, background=0)
		mask = np.zeros(thresh.shape, dtype="uint8")

		# loop over the unique components
		for label in np.unique(labels):
			# if this is the background label, ignore it
			if label == 0:
				continue

			# otherwise, construct the label mask and count the
			# number of pixels
			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			numPixels = cv2.countNonZero(labelMask)

			# if the number of pixels in the component is sufficiently
			# large, then add it to our mask of "large blobs"
			if numPixels > 300:
				mask = cv2.add(mask, labelMask)

		return mask

	# ----------------------------------------------------------------------------
	#                         mse and ssim
	# ----------------------------------------------------------------------------
	def mse(imageA, imageB):
		# the 'Mean Squared Error' between the two images is the
		# sum of the squared difference between the two images;
		# NOTE: the two images must have the same dimension
		err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
		err /= float(imageA.shape[0] * imageA.shape[1])

		# return the MSE, the lower the error, the more "similar"
		# the two images are
		return err


	# ----------------------------------------------------------------------------
	#                        optical flow mask calcuation
	# ----------------------------------------------------------------------------

	def opticalFlowMask(prevFullFrame, curtFullFrame, startX, endX, startY, endY):
		prevGray = cv2.cvtColor(prevFullFrame,cv2.COLOR_BGR2GRAY)
		curtGray = cv2.cvtColor(curtFullFrame,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prevGray, curtGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

		hsv = np.zeros_like(prevFullFrame)
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,1] = 255
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

		opticalFlow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		croppedGrayOpticalFlow = cv2.cvtColor(opticalFlow[startX:endX, startY:endY, :], cv2.COLOR_BGR2GRAY)


		croppedOpticalFlowMask = np.ones(croppedGrayOpticalFlow.shape[:2],dtype="uint8")
		opticalThreshold = croppedGrayOpticalFlow.mean()  # thresholding
		opticalFlowMaskInd = np.where(croppedGrayOpticalFlow > opticalThreshold)
		croppedOpticalFlowMask[opticalFlowMaskInd] = 0

		#return croppedGrayOpticalFlow
		return croppedOpticalFlowMask


	# ------------------------------------------------------------------
	#                  Eye detection
	# ------------------------------------------------------------------
	def eyeDetect(frame, detector, predictor):

		# define two constants, one for the eye aspect ratio to indicate
		# blink and then a second constant for the number of consecutive
		# frames the eye must be below the threshold for to set off the
		# alarm
		EYE_AR_THRESH = 0.15
		EYE_AR_CONSEC_FRAMES = 48

		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# Convert the cropped face to grayscale channel
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# eye close flag
		eyeCloseFlag = 0

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = MyClass.eye_aspect_ratio(leftEye)
			rightEAR = MyClass.eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)

			if ear < EYE_AR_THRESH:
				eyeCloseFlag = 1


		return leftEyeHull, rightEyeHull, eyeCloseFlag

	def eye_aspect_ratio(eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)

		# return the eye aspect ratio
		return ear


	# ------------------------------------------------------------------
	#                  main function
	# ------------------------------------------------------------------

	def my_main(filename, haha):

		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		# ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
		# ap.add_argument("-m", "--model", required=True,	help="path to Caffe pre-trained model")
		ap.add_argument("-p", "--prototxt", type=str, default="deploy.prototxt.txt", help="path to Caffe 'deploy' prototxt file")
		ap.add_argument("-m", "--model", type=str, default="res10_300x300_ssd_iter_140000.caffemodel", help="path to Caffe pre-trained model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
		args = vars(ap.parse_args())

		# load our serialized model from disk
		print("[INFO] loading face model...")
		net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
		#net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		print("[INFO] loading eye detection model")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		# ---------------------------------------------------
		# parameters preparation
		lowerSkin1 = np.array([0, 48, 80], dtype = "uint8")
		upperSkin1 = np.array([20, 255, 255], dtype = "uint8")

		lowerSkin2 = np.array([0, 0.23 * 255, 50], dtype = "uint8")
		upperSkin2 = np.array([50, 0.68 * 255, 255], dtype = "uint8")

		lowerSkin = lowerSkin1
		upperSkin = upperSkin1

		# ------------------------------------------------------------
		#                    Connected to Database
		conn = sqlite3.connect("db\\syncVideo.db")
		cur = conn.cursor()
		cur.execute("SELECT frameID FROM FRAME")
		resultSet = cur.fetchall()
		frameInfo = []
		for p in range(len(resultSet)):
			frameInfo.append(resultSet[p][0])
		print(frameInfo)

		# ------------------------------------------------------------
		#                    Video Post-processing
		# ------------------------------------------------------------

		# Read the recorded face video
		print("[INFO] starting video processing...")
		modelSelect = 0
		vs = cv2.VideoCapture('testVideo\\' + filename + '.mp4')
		totalFrameCnt = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
		# totalFrameCnt = int(1000)
		w = int(vs.get(3))
		h = int(vs.get(4))

		# loop over the frames from the video stream
		histList = []
		fig = plt.figure()
		plt.ion()

		for frameCnt in tqdm(range(totalFrameCnt)):
			# grab the frame from thevideo stream
			ret, frame = vs.read()

			# Select between Eye and Face/skin detection: Dns
			dns = 0
			eyeCloseList = []
			if dns == 1:
				# ------------------------------------------------
				#           Eye Detection  -- high ambient light
				# ------------------------------------------------
				leftEyeHull, rightEyeHull, eyeCloseFlag = MyClass.eyeDetect(frame, detector, predictor)
				eyeCloseList.append(frameCnt)
				# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				leftEyeRect = []
				for p in range(len(leftEyeHull)):
					if p == 0:
						xMax = leftEyeHull[p][0][0]
						xMin = leftEyeHull[p][0][0]
						yMax = leftEyeHull[p][0][1]
						yMin = leftEyeHull[p][0][1]
					else:
						if leftEyeHull[p][0][0] >= xMax:
							xMax = leftEyeHull[p][0][0]
						if leftEyeHull[p][0][0] < xMin:
							xMin = leftEyeHull[p][0][0]

						if leftEyeHull[p][0][1] >= yMax:
							yMax = leftEyeHull[p][0][1]
						if leftEyeHull[p][0][1] < yMin:
							yMin = leftEyeHull[p][0][1]
				leftEyeRect = [xMin, xMax, yMin, yMax]


				rightEyeRect = []
				for p in range(len(rightEyeHull)):
					if p == 0:
						xMax = rightEyeHull[p][0][0]
						xMin = rightEyeHull[p][0][0]
						yMax = rightEyeHull[p][0][1]
						yMin = rightEyeHull[p][0][1]
					else:
						if rightEyeHull[p][0][0] >= xMax:
							xMax = rightEyeHull[p][0][0]
						if rightEyeHull[p][0][0] < xMin:
							xMin = rightEyeHull[p][0][0]

						if rightEyeHull[p][0][1] >= yMax:
							yMax = rightEyeHull[p][0][1]
						if rightEyeHull[p][0][1] < yMin:
							yMin = rightEyeHull[p][0][1]
				rightEyeRect = [xMin, xMax, yMin, yMax]


				eyeMask = np.zeros(frame.shape[:2])
				cv2.fillConvexPoly(eyeMask, leftEyeHull, 1)
				eyeMask = eyeMask.astype(np.bool)
				croppedLeftEye = np.zeros_like(frame)
				croppedLeftEye[eyeMask] = frame[eyeMask]
				leftEye = croppedLeftEye[leftEyeRect[2]:leftEyeRect[3], leftEyeRect[0]:leftEyeRect[1]]

				eyeMask = np.zeros(frame.shape[:2])
				cv2.fillConvexPoly(eyeMask, rightEyeHull, 1)
				eyeMask = eyeMask.astype(np.bool)
				croppedRightEye = np.zeros_like(frame)
				croppedRightEye[eyeMask] = frame[eyeMask]
				rightEye = croppedRightEye[rightEyeRect[2]:rightEyeRect[3], rightEyeRect[0]:rightEyeRect[1]]

				#rightEye = cv2.resize(rightEye, leftEye.shape[:2])
				#skin = np.concatenate((leftEye, rightEye), axis=0)

				skin = leftEye

			else:
				# ------------------------------------------------
				#           Face Detection
				# ------------------------------------------------
				# grab the frame dimensions and convert it to a blob
				blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

				# pass the blob through the network and obtain the detections and predictions
				net.setInput(blob)
				detections = net.forward()

				# loop over the detections
				for i in range(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated with the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by ensuring the `confidence` is
					# greater than the minimum confidence
					if confidence < args["confidence"]:
						continue

					# compute the (x, y)-coordinates of the bounding box for the object
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# crop face from frame
					croppedFace = frame[startY:endY, startX:endX]
					croppedFace = ndimage.median_filter(croppedFace, 3)   # median filter


					# crop chin .... for bright environment
					faceShape = croppedFace.shape
					croppedChin = croppedFace[int(5*faceShape[0]/6):, :, :]

					hsv1 = cv2.cvtColor(croppedFace, cv2.COLOR_BGR2HSV)
					hsv1[:,:,0] = ((255.0/179.0)*hsv1[:,:,0]).astype('uint8')
					hsv2 = cv2.cvtColor(hsv1, cv2.COLOR_RGB2HSV)
					red = hsv2[:,:,0]
					_,outMask = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
					chinShadow = cv2.bitwise_and(croppedFace, croppedFace, mask=outMask)


					# ------------------------------------------------
					#           Skin Detection
					# ------------------------------------------------
					# convert it to the HSV color space,
					# and determine the HSV pixel intensities that fall into
					# the speicifed upper and lower boundaries
					convertedFace = cv2.cvtColor(croppedFace, cv2.COLOR_BGR2HSV)
					skinMask = cv2.inRange(convertedFace, lowerSkin, upperSkin)

					# apply a series of erosions and dilations to the mask
					# using an elliptical kernel
					kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
					skinMask = cv2.erode(skinMask, kernel, iterations = 2)
					skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

					# blur the mask to help remove noise, then apply the mask to the frame
					skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)


					# # Inverse select
					# skinMask[np.where(skinMask == 0)] = 1
					# skinMask[np.where(skinMask != 0)] = 0
					# skin = cv2.bitwise_and(croppedFace, croppedFace, mask=skinMask)

					# calculate how many skin pixels in face region
					skinInd = np.where(skinMask!=0)
					skinSize = skinInd[0].size

					# eliminate incorrect face region
					faceSize = (endY-startY)*(endX-startX)

					if skinSize < 0.3*faceSize:
						print("not enough pixels.", detections.shape[2])

						# Switch the skin color model:
						if sum(lowerSkin-lowerSkin1) == 0:
							lowerSkin = lowerSkin2
							upperSkin = upperSkin2
						else:
							lowerSkin = lowerSkin1
							upperSkin = upperSkin1

						convertedFace = cv2.cvtColor(croppedFace, cv2.COLOR_BGR2HSV)
						skinMask = cv2.inRange(convertedFace, lowerSkin, upperSkin)
						kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
						skinMask = cv2.erode(skinMask, kernel, iterations = 2)
						skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
						skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
						skin = cv2.bitwise_and(croppedFace, croppedFace, mask=skinMask)

						skinInd = np.where(skinMask!=0)
						skinSize = skinInd[0].size

						if skinSize < 0.3*faceSize:
							skin = croppedFace
					else:
						saveCroppedFace = croppedFace.copy()
						# Place mask on skin (convertedFace seems better than croppedFace)

						skin = cv2.bitwise_and(croppedFace, croppedFace, mask=skinMask)
						skin = cv2.bitwise_and(convertedFace, convertedFace, mask=skinMask)

					# skin = chinShadow
					cv2.imshow("cropped", skin)
					cv2.moveWindow("cropped", 0, 0)
					cv2.waitKey(1)

					if frameCnt == 78:
						cv2.imwrite( "2.png", skin);
						cv2.waitKey(5000)
					elif frameCnt == 70:
						cv2.imwrite( "1.png", skin);
						cv2.waitKey(5000)
					else:
						cv2.waitKey(1)


			# show the output frame; save frame to directory (optional)


			# # -------------------------------------------------
			# #              Color Model
			# # -------------------------------------------------
			# if (frameCnt <= 1000):
			# 	blackThreshold = 0
			# 	binNum = [32,32,32]
			# 	prevFrame = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
			# 	prevMask = np.zeros(prevFrame.shape[:2],dtype="uint8")
			# 	prevMaskInd = np.where((prevFrame[:,:,0] > blackThreshold) | (prevFrame[:,:,1] > blackThreshold) | (prevFrame[:,:,2] > blackThreshold))
			# 	prevMask[prevMaskInd] = 1


			# 	color = ('b','g','r')
			# 	for i,col in enumerate(color):
			# 	    prevHist = cv2.calcHist([prevFrame],[i],prevMask,[256],[0,256])
			# 	    plt.plot(prevHist,color = col)
			# 	plt.show()
			# 	plt.pause(0.05)
			# 	plt.clf()

			# 	continue
			# break

			# -------------------------------------------------
			#   Inter-Frace Comparision for Flash Detection
			# -------------------------------------------------

			# 1. calculte histogram distance
			# 2. derive optical flow
			if frameCnt == 0:
				prevFrame = skin.copy()
				prevFullFrame = frame.copy()
			else:
				curtFrame = skin
				curtFullFrame = frame.copy()

				# ---------------------------------------------
				#           derive optical flow mask
				# ---------------------------------------------
				#croppedOpticalFlowMask = opticalFlowMask(prevFullFrame, curtFullFrame, startY, endY, startX, endX)
				#curtFrame = cv2.bitwise_and(curtFrame, curtFrame, mask=croppedOpticalFlowMask)

				# --------------------------------------------
				#       face region histgram calculation
				# --------------------------------------------
				curtH, curtW, curtC = curtFrame.shape
				resizedPrevFrame = cv2.resize(prevFrame, (curtW, curtH), interpolation = cv2.INTER_AREA)

				# # Optional:  pick special pixels
				# sv = 1              # satuation: 1 or brightnss: 2 channel
				# prevSkinMask = senstiveSkin(prevFrame, sv, 0.0)
				# curtSkinMask = senstiveSkin(resizedCurtFrame, sv, 0.0)
				# prevFrame = cv2.bitwise_and(prevFrame, prevFrame, mask=prevSkinMask)
				# resizedCurtFrame = cv2.bitwise_and(resizedCurtFrame, resizedCurtFrame, mask=curtSkinMask)

				# #remove suspicious noise
				# noiseInd = np.where(np.absolute(curtFrame - resizedPrevFrame) > 100)
				# curtFrame[noiseInd] = 0
				# resizedPrevFrame[noiseInd] = 0

				# Configure hist set and method
				SET_IND = 1
				METHOD_IND = 1
				binSize = 64
				histDiff, prevHist, curtHist = MyClass.histCompare(resizedPrevFrame, curtFrame, binSize, SET_IND, METHOD_IND, modelSelect)
				histList.append(histDiff)

				# ---------------------------
				# Troubleshooting observation
				# ---------------------------

				# frameDiff = curtFrame - resizedPrevFrame
				# frameBoth = np.hstack((resizedPrevFrame,curtFrame))
				# cv2.imshow("cropped", frameDiff)
				# cv2.moveWindow("cropped", 0, 0)


				# print(prevHist.shape)
				# plt.plot(prevHist)
				# plt.show()

				# diffHist = cv2.calcHist([curtFrame], [0, 1, 2], None, [16,16,16], [0, 256, 0, 256, 0, 256])
				# cv2.normalize(diffHist, diffHist)
				# diffHist = diffHist.flatten()
				# plt.plot(diffHist)
				# plt.show()
				# plt.pause(0.05)
				# plt.clf()


				# f = np.fft.fft2(curtFrame[:,:,0])
				# fshift = np.fft.fftshift(f)
				# magnitude_spectrum = 20*np.log(np.abs(fshift))
				# plt.imshow(magnitude_spectrum,  cmap = 'gray')
				# plt.show()
				# plt.pause(0.05)
				# plt.clf()

				# if histDiff > 1.5:
				# 	print(frameCnt)
				# 	cv2.waitKey(1000)

				# else:
					# diffHist = cv2.calcHist([frameDiff], [0, 1, 2], None, [16,16,16], [0, 256, 0, 256, 0, 256])
					# cv2.normalize(diffHist, diffHist)
					# diffHist = diffHist.flatten()
					# plt.plot(diffHist)
					# plt.show()

					# cv2.waitKey(100)


				prevFrame = curtFrame.copy()
				prevFullFrame = curtFullFrame.copy()


		# histList = np.delete(histList, eyeCloseList).tolist()
		# do a bit of cleanup
		vs.release()
		cv2.destroyAllWindows()

		# close db
		conn.close()

		# ------------------------------------------------------------
		#                    Data Post-processing
		# ------------------------------------------------------------

		# # calculate rolling mean and variance
		# movMean = pd.rolling_mean(np.asarray(histList), 11)
		# histList = movMean
		# movStd = pd.rolling_std(np.asarray(histList), 5)
		# histList = movStd

		if modelSelect == 1:
			histList = ndimage.median_filter(histList, 3)   # median filter

		# Plot hist variation
		fig = plt.figure()
		plt.plot(histList, 'r--')
		plt.savefig('hist.png')
		plt.show()
		#plt.close()
		haha.put(item=True)
		print("done.")

