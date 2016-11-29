# USAGE
# python cam.py --face cascades/haarcascade_frontalface_default.xml
# python cam.py --face cascades/haarcascade_frontalface_default.xml --video video/adrian_face.mov

# import the necessary packages
from face_dec_lib.facedetector import FaceDetector
from face_dec_lib import imutils
import argparse
import datetime
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")
ap.add_argument("-pf", "--profile_face",
				help = "path to where the profile face cascade resides")
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# construct the face detector
fd = FaceDetector(args["face"])
if args["profile_face"]:
	pfd = FaceDetector(args["profile_face"])

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width = 500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Reset the flags as default
	detected_front = True
	detected = True

	frameClone = frame.copy()

	# detect faces in the image and then clone the frame
	# so that we can draw on it
	faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
	if len(faceRects) == 0:
		detected_front = False
		print_text = "No face!"
	else:
		print_text = "Front face detected!"
		# loop over the face bounding boxes and draw them
		for (fX, fY, fW, fH) in faceRects:
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

	if args["profile_face"] and not detected_front:
		faceRects_pf = pfd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
		if len(faceRects_pf) == 0:
			detected = False
			print_text = "No face!"
		else:
			print_text = "Profile face detected!"
			# loop over the face bounding boxes and draw them
			for (fX, fY, fW, fH) in faceRects_pf:
				cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 255), 2) 
	
	# Add the date and time of the current pic
	cv2.putText(frameClone, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
	cv2.putText(frameClone, "Status: {}".format(print_text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# show our detected faces
	cv2.imshow("Face", frameClone)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
