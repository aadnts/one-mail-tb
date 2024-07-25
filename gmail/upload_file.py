from flask import Flask, request, render_template
from flask_cors import CORS
import token_generator
import os


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'credentials/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], "credentials.json"))
        token_generator.generate_token()
        return 'File successfully uploaded', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
