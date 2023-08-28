from flask import Flask,render_template,request,jsonify,Response
import detect
import cv2
import base64
from torchvision import transforms
from io import BytesIO
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
from flask_socketio import SocketIO, send,emit
import io
def loadYoloTorchModel(path):
    return torch.hub.load('.', 'custom', path=path, source='local')
def loadUltraYoloTorchModel(path):
    return YOLO(path)

app = Flask(__name__)
socketio = SocketIO(app)
model = loadYoloTorchModel('./best.pt')
# model = loadUltraYoloTorchModel('./best_detect_v8_epoch80.pt')
client_list = []

@app.route('/')
def home () :
    return render_template('index.html')

def detectYoloTorchModel(model,image):
    result = model(image)
    result.render()
    detectList = result.pandas().xyxy[0].to_json(orient='records')
    buffered = BytesIO()
    im_base64 = Image.fromarray(result.ims[0])
    im_base64.save(buffered, format="JPEG")
    encode = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encode,detectList

def detectUltraTorchModel(model,image):
    result = model(image)
    detectList = result[0].tojson()
    buffered = BytesIO()
    im_base64 = Image.fromarray(result[0].orig_img)
    im_base64.save(buffered, format="JPEG")
    encode = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encode,detectList
@app.route('/upload',methods=['POST'])
def uploadImage () :
    image = request.files['image']
    im_bytes = image.read()
    im = Image.open(io.BytesIO(im_bytes))
    encode,result = detectYoloTorchModel(model,im)
    # encode,result = detectUltraTorchModel(model,im)
    return jsonify({'resultImage':encode,'message':str(result)})

@socketio.on('connect', namespace='/live')
def socket_connect():
    print('Client wants to connect.')
    emit('response', {'data': 'OK'})

@socketio.on('disconnect', namespace='/live')
def socket_disconnect():
    print('Client disconnected')


@socketio.on('event', namespace='/live')
def socket_message(message):
    emit('response',
         {'data': message['data']})
    print(message['data'])


@socketio.on('livevideo', namespace='/live')
def socket_live(message):
    image = message['data']
    imgdata = base64.b64decode(image.split('base64,')[1])
    im = Image.open(io.BytesIO(imgdata))
    ret,message = detectYoloTorchModel(model,im)
    # ret,message = detectUltraTorchModel(model,im)
    emit('ret_stream',{'data':ret})
if __name__ == '__main__':
    app.run(port=8080,debug=True)
    cv2.destroyAllWindows()


