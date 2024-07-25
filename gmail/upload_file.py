from flask import Flask, request, render_template
from flask_cors import CORS
import orchestrator
import token_generator
import os
import os
import pickle
from flask import Flask, request, jsonify, redirect, session, url_for
from google_auth_oauthlib.flow import Flow
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, resources={r"/*": {"origins": "http://104.248.236.94:5173"}})
app.secret_key = "!ZiF7r2@WbDwB3"

CREDENTIALS_FOLDER = 'credentials/'
os.makedirs(CREDENTIALS_FOLDER, exist_ok=True)
app.config['CREDENTIALS_FOLDER'] = CREDENTIALS_FOLDER

OCR_FOLDER = 'ocr/'
os.makedirs(OCR_FOLDER, exist_ok=True)
app.config['OCR_FOLDER'] = OCR_FOLDER

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
UPLOAD_FOLDER = '/root/one-mail-tb/gmail/credentials'
CLIENT_SECRETS_FILE = os.path.join(UPLOAD_FOLDER, "credentials.json")
TOKEN_PICKLE_FILE = os.path.join(UPLOAD_FOLDER, "token.pickle")
ALLOWED_EXTENSIONS = {'json'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_credentials', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        try:
            filename = 'credentials.json'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(f"File {filename} saved successfully")  # Detailed log
            return jsonify({'message': 'File successfully uploaded'}), 200
        except Exception as e:
            print(f"Error saving file: {e}")  # Detailed log
            return jsonify({'error': str(e)}), 500
    else:
        print("Invalid file type")  # Detailed log
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate_token', methods=['GET'])
def generate_token():
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return jsonify({'error': 'Credentials file not found'}), 400
    
    try:
        flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
        flow.redirect_uri = url_for('oauth2callback', _external=True)

        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true')

        session['state'] = state
        print(f"Authorization URL generated: {authorization_url}")  # Detailed log
        return jsonify({'authorization_url': authorization_url})
    except Exception as e:
        print(f"Error generating token: {e}")  # Detailed log
        return jsonify({'error': str(e)}), 500

@app.route('/oauth2callback', methods=['GET'])
def oauth2callback():
    state = session['state']
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)

    creds = flow.credentials
    with open(TOKEN_PICKLE_FILE, 'wb') as token:
        pickle.dump(creds, token)

    return redirect('/token_saved')

@app.route('/token_saved', methods=['GET'])
def token_saved():
    return 'Token saved successfully.'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_filee():

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
    
