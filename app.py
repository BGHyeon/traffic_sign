from flask import Flask,render_template,request,jsonify
import detect
from PIL import Image

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
    rawBytes = BytesIO()

    # 2차원 배열의 type은 uint8이여야 인코딩 가능
    img_buffer = Image.fromarray(proceedImage[0].astype('uint8'))

    img_buffer.save(rawBytes, 'PNG')
    rawBytes.seek(0)
    base64_img = base64.b64encode(rawBytes.read()).decode('utf-8')
    # encodeImage = base64.b64encode(Image.fromarray(proceedImage[0]).read()).decode('utf-8')
    return jsonify({'resultImage':base64_img,'message':'success'})
if __name__ == '__main__':
    app.run(port=8080,debug=True)