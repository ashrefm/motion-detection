import cv2
import numpy as np
from datetime import datetime
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--video_file', default='data/sample.avi',
                    help="Path for the video file conaining the sequence")



def diffImg(t0, t1, t2):				# Function to calculate difference between images
	d1 = cv2.absdiff(t2, t1)
	d2 = cv2.absdiff(t1, t0)
	return cv2.bitwise_and(d1, d2)

threshold = 1000000						# Threshold for triggering motion detection


if __name__ == '__main__':

	args = parser.parse_args()

	cap = cv2.VideoCapture(args.video_file)

	winName = "Motion Detector"				# Application name
	cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(winName, 850, 640)

	# Read three images first
	t_minus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
	t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
	t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

	# Use a time check so we only take 1 pic per sec
	timeCheck = datetime.now().strftime('%Ss')

	counter = 0 # counts the number of detected rims exported on disk
	while cap.isOpened():

		ret, frame = cap.read()

		if ret == True:

			if cv2.countNonZero(diffImg(t_minus, t, t_plus)) > threshold and timeCheck != datetime.now().strftime('%Ss'):
				export_frame = True
			else:
				export_frame = False

			img= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			img = cv2.GaussianBlur(img, (21,21), cv2.BORDER_DEFAULT)
			#all_circs = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, 120, param1=50, param2=30, minRadius=60, maxRadius=1000)
			all_circs = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=0.9, minDist=120, minRadius=300)

			img_contour = frame.copy()

			detected = False # wether or not a rim appears in the frame
			if all_circs is not None:
				detected = True
				# highlight the contour and center of the rim
				all_circs_rounded = np.uint16(np.around(all_circs))
				for i in all_circs_rounded[0, :]:
					cv2.circle(img_contour, (i[0], i[1]), i[2], (0, 255, 0), 10)
					cv2.circle(img_contour, (i[0], i[1]), 10, (255, 0, 0), 10)

			if export_frame == True and detected == True:
				counter +=1
				print('Saving new frame %d' % counter)
				# save original frame
				cv2.imwrite(os.path.join('output', datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '_original.jpg'), frame)
				# save the detection frame with contour
				#cv2.imwrite(os.path.join('output', datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'), img_contour)

			cv2.imshow(winName, img_contour)
			if cv2.waitKey(1) & 0xFF == ord('q'):
			    break

			timeCheck = datetime.now().strftime('%Ss')

			# Read next image
			t_minus = t
			t = t_plus
			t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

		else:
			break


	cap.release()
	cv2.destroyAllWindows()

