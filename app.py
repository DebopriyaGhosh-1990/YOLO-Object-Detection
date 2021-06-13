from flask import Flask,render_template, request,Response, redirect,url_for
import cv2
import numpy as np
import readchar, keyboard


app=Flask(__name__)

whTar = 320
confThreshold = 0.5
NMSThreshold = 0.3

classesFile = './models/coco.names'
classesName = []
f = open(classesFile, 'rt')
classesName = f.read().rstrip('\n').split('\n')
#print(classesName)

modelConfiguration = './models/yolov3-320.cfg'
modelWeights = './models/yolov3-320.weights'

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
        text = classesName[classIds[i]].upper() +': '+str(int(Confs[i]*100))+'%'
        text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,0.9,3)
        text_w, text_h = text_size[0]
        cv2.rectangle(img, (x,y),(x+w,y+h),(45,255,255),2)
        cv2.rectangle(img, (x, y), (x+text_w, y-50), (45,255,255), -1)
        cv2.putText(img,f'{classesName[classIds[i]].upper()}: {int(Confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)




def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error opening Camera')

    while (cap.isOpened()):
        #cap.set(3,640)
        #cap.set(4,480)
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (whTar, whTar), [0, 0, 0], 1, crop=False)
            net.setInput(blob)

            layerNames = net.getLayerNames()
            # print(layerNames)
            outputLayerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            # print(net.getUnconnectedOutLayers())
            # print(outputLayerNames)
            output = net.forward(outputLayerNames)
            # print(output[0].shape)
            # print(output[1].shape)
            # print(output[2].shape)
            # print(output[0][0])

            findObjects(output, img)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/', methods=['GET'])
def YOLO_Home():
    return render_template('index.html')

@app.route('/YOLO', methods=['GET','POST'])
def YOLO():
    if request.method == 'POST':
        return render_template('DetectMe.html')
    else:
        cv2.VideoCapture(0).release()
        return render_template('index.html')

@app.route('/ObjectDetect', methods=['GET','POST'])
def ObjectDetect():
    if request.method == 'GET':

        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        cv2.VideoCapture(0).release()
        return render_template('index.html')

@app.route('/back', methods=['GET','POST'])
def back():

        cv2.VideoCapture(0).release()
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host = '0.0.0.0',debug=True)