from flask import Flask,render_template,request,jsonify,Response
import detect
import cv2
import base64
from torchvision import transforms
from io import BytesIO
import torch
from PIL import Image
import numpy as np
from flask_socketio import SocketIO, send,emit
import io

app = Flask(__name__)
socketio = SocketIO(app)
model = detect.load_model('./best.pt')
# model = torch.hub.load('.', 'custom', path='./best.pt', source='local')

modelVideo = torch.hub.load('.', 'custom', path='./best.pt', source='local')
client_list = []
@app.route('/')
def home () :
    return render_template('index.html')


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            camera.release()
            break
        else:
            frame = frame[:, :, [2, 1, 0]]
            frame = Image.fromarray(frame)
            detected = modelVideo(frame)
            r_img = detected.render()
            rawBytes = BytesIO()

            # 2차원 배열의 type은 uint8이여야 인코딩 가능
            img_buffer = Image.fromarray(r_img[0])

            img_buffer.save(rawBytes, 'PNG')
            rawBytes.seek(0)
            # retval, buffer = cv2.imencode('.jpg', r_img)
            frame = rawBytes.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()



@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/upload',methods=['POST'])
def uploadImage () :
    image = request.files['image']
    image.save('tmp.jpg')
    proceedImage = detect.detect(source='./tmp.jpg', model= model)
    retval, buffer = cv2.imencode('.jpg', proceedImage[0][0])
    encodeImage = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'resultImage':encodeImage,'message':str(proceedImage[0][1])})

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
    results = modelVideo(im)
    results.render()
    buffered = BytesIO()
    im_base64 = Image.fromarray(results.ims[0])
    im_base64.save(buffered, format="JPEG")
    ret = base64.b64encode(buffered.getvalue()).decode('utf-8')
    emit('ret_stream',{'data':ret})
if __name__ == '__main__':
    app.run(port=8080,debug=True)
    cv2.destroyAllWindows()


