# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help = "Path to input image")
ap.add_argument("-d", "--detector", required=True,
	help = "Path to face detector")
ap.add_argument("-m", "--embedding_model", required =True,
	help = "Path to embedding extractor")
ap.add_argument("-r", "--recognizer", required =True,
	help = "Path to face recognizer")
ap.add_argument("-l", "--le", required =True,
	help = "Path to label encoder")
ap.add_argument("-c", "--confidence", type = float, default =0.5,
	help = "Confidence threshold to filter weak detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"],"rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 600)
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB = False, crop =False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(blob)
detections = detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0,0,i,3:7] * np.array([w,h,w,h])
		startX, startY, endX, endY = box.astype("int")
		
		# extract the face ROI
		face = image[startY:endY, startX:endX]
		fH, fW = face.shape[:2]
		
		if fH<20 or fW<20:
			continue
		
		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0,0,0), swapRB =True, crop = False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		print(preds)
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# draw the bounding box of the face along with the associated
		# probability
		text ="{}: {:.2f}%".format(name, proba*100)
		y = startY-10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
fig, axs = plt.subplots(1,1, figsize=(15,10))
fig.tight_layout()

axs[0].imshow(image)
axs[0].set_title('Image')
cv2.imshow("Image", image)
cv2.waitKey(0)
