from flask import Flask,render_template,request,jsonify
import base64
import cv2 
import numpy as np
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
net = cv2.dnn.readNet(ROOT_DIR+"/yolov3-tiny.weights",ROOT_DIR+ "/v3-all.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
app = Flask(__name__)
def imageDetection (image):
    img = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    return img
@app.route('/')
def home () :
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def uploadImage () : 
    image = request.files['image']
    img = imageDetection(image.read())
    encodeImage = base64.b64encode(img).decode('utf-8')
    return jsonify({'resultImage':encodeImage,'message':'success'})
if __name__ == '__main__':
    app.run(port=8080,debug=True)