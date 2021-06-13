import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whTar = 320
confThreshold = 0.5
NMSThreshold = 0.3

classesFile = 'coco.names'
classesName = []
f = open(classesFile,'rt')
classesName = f.read().rstrip('\n').split('\n')
print(classesName)

modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    Confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classIdx = np.argmax(scores)
            conf = scores[classIdx]

            if conf > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classIdx)
                Confs.append(float(conf))
    #print(bbox)
    indices = cv2.dnn.NMSBoxes(bbox,Confs,confThreshold,NMSThreshold)
    #print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classesName[classIds[i]].upper()} {int(Confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)



while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whTar,whTar),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    outputLayerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(net.getUnconnectedOutLayers())
    #print(outputLayerNames)
    output = net.forward(outputLayerNames)
    #print(output[0].shape)
    #print(output[1].shape)
    #print(output[2].shape)
    #print(output[0][0])

    findObjects(output,img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)