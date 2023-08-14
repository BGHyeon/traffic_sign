from flask import Flask,render_template,request,jsonify
import base64
app = Flask(__name__)

@app.route('/')
def home () :
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def uploadImage () : 
    image = request.files['image']
    encodeImage = base64.b64encode(image.read()).decode('utf-8')

    return jsonify({'resultImage':encodeImage,'message':'success'})
if __name__ == '__main__':
    app.run(port=8080,debug=True)