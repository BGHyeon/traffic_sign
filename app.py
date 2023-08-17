import cv2
from flask import Flask,render_template,request,jsonify
import base64
import ex1
app = Flask(__name__)
model = cv2.dnn.readNet("model/yolov3-tiny.cfg","model/yolov3-tiny.weights")
@app.route('/')
def home () :
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def uploadImage () : 
    image = request.files['image']
    detectedImage,detectedLabel = ex1.detection(image,model)

    retval, buffer = cv2.imencode('.jpg', detectedImage)
    encodeImage = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'resultImage':encodeImage,'message':str(detectedLabel)})
if __name__ == '__main__':
    app.run(port=8080,debug=True)