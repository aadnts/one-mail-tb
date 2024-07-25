from flask import Flask, request, render_template
from flask_cors import CORS
import orchestrator
import token_generator
import os


app = Flask(__name__)
CORS(app)

CREDENTIALS_FOLDER = 'credentials/'
os.makedirs(CREDENTIALS_FOLDER, exist_ok=True)
app.config['CREDENTIALS_FOLDER'] = CREDENTIALS_FOLDER

OCR_FOLDER = 'ocr/'
os.makedirs(OCR_FOLDER, exist_ok=True)
app.config['OCR_FOLDER'] = OCR_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        file.save(os.path.join(app.config['CREDENTIALS_FOLDER'], "credentials.json"))
        token_generator.generate_token()
        return 'File successfully uploaded', 200

@app.route('/ocr', methods=['POST'])
def ocr():

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        file.save(os.path.join(app.config['OCR_FOLDER'], file.filename))
        print('File successfully uploaded')
        orchestrator.send_files()
        return 'File successfully processed', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
