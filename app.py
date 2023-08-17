from flask import Flask,render_template,request,jsonify
import detect
import cv2
import base64
from io import BytesIO
app = Flask(__name__)
@app.route('/')
def home () :
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def uploadImage () :
    image = request.files['image']
    image.save('tmp.jpg')
    proceedImage = detect.detect(source='./tmp.jpg',weights='./best.pt')
    retval, buffer = cv2.imencode('.jpg', proceedImage[0][0])
    encodeImage = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'resultImage':encodeImage,'message':str(proceedImage[0][1])})
if __name__ == '__main__':
    app.run(port=8080,debug=True)