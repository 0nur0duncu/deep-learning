# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.7,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")    
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

net = cv2.dnn.readNetFromDarknet("yolov3-tiny_obj.cfg", "yolov3-tiny_obj_final.weights")

ln = net.getLayerNames()
print("ln",net)
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
#vs = cv2.VideoCapture("saricember.mp4")
vs = cv2.VideoCapture("saricember.mp4")
writer = None
video_parameters = {"total_frame": int(vs.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "FPS": int(vs.get(cv2.CAP_PROP_FPS)),
                    "frame_width":int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "frame_height": int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))}


W = video_parameters['frame_width']
H = video_parameters['frame_height'] 

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (ret, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not ret:
        break

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

# loop over each of the layer outputs
        # loop over each of the detections
    for detection in layerOutputs[0]:
        # extract the class ID and confidence (i.e., probability)
        # of the current object detection
        classID = 0
        confidence = detection[5]
        
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to
            # the size of the image, keeping in mind that YOLO
            # actually returns the center (x, y)-coordinates of
            # the bounding box followed by the boxes' width and
            # height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top
            # and and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])
    # ensure at least one detection exists
    if len(idxs) > 0:   
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (centerX - 5, centerY - 5), (centerX + 5, centerY + 5), (0, 128, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.putText(frame, "FPS: " + str(int(vs.get(cv2.CAP_PROP_FPS))), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vs.release()
cv2.destroyAllWindows()