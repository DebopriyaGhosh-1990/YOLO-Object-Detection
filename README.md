# YOLO-Object-Detection
Real time Object Detection using YOLOV3 Algorithm


Comparison to Other Detectors
YOLOv3 is extremely fast and accurate.Moreover, you can easily tradeoff between speed and accuracy simply by changing the size of the model, no retraining required!

How It Works:
Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.

We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

This model has several advantages over classifier-based systems. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast, more than 1000x faster than R-CNN and 100x faster than Fast R-CNN.

What's New in Version 3?
YOLOv3 uses a few tricks to improve training and increase performance, including: multi-scale predictions.

For the object detection below pretrained model has been used:
YOLOv3-320 COCO trainval test-dev 51.5 38.97 Bn 45
This is an POC for real time object detection using YOLOV3 end to end and model deployment using HEROKU PAAS cloud platform
Download the below config and weights file for pretrained YOLO model 320:

YOLOv3-320 weights file : https://pjreddie.com/media/files/yolov3.weights

YOLOv3-320 config file : https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

This repositoty contains below files and folders:
1. app.py file for deployment and detection
2. models folder containing weights and config file for YOLO-320 architecture
3. static file for images and css
4. templates for html files
