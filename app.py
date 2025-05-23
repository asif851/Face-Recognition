from flask import Flask, render_template, request
from detect import detect_face_and_predict
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    name, confidence = detect_face_and_predict(file_path)
    confidence_percent = round(confidence * 100, 2)

    return render_template('result.html', name=name, confidence=confidence_percent, img_path=file_path)



if __name__ == '__main__':
    app.run(debug=True,port=3000)
