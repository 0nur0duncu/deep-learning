import numpy as np
cimport numpy as np

def process_yolo_detections(np.ndarray[np.float64_t, ndim=2] layerOutputs,
                            dict args, int W, int H):
    cdef list boxes = []
    cdef list confidences = []
    cdef list classIDs = []

    cdef int classID
    cdef double confidence
    cdef np.ndarray[np.float64_t, ndim=1] box

    for detection in layerOutputs:
        classID = 0
        confidence = detection[5]

        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            centerX = int(box[0])
            centerY = int(box[1])
            width = int(box[2])
            height = int(box[3])

            x = centerX - (width // 2)
            y = centerY - (height // 2)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            classIDs.append(classID)

    return boxes, confidences, classIDs
