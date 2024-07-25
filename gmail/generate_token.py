import os
import pickle
from flask import Flask, redirect, request, session, url_for, jsonify
from google_auth_oauthlib.flow import Flow

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CLIENT_SECRETS_FILE = "/root/one-mail-tb/gmail/credentials/credentials.json"
TOKEN_PICKLE_FILE = "/root/one-mail-tb/gmail/credentials/token.pickle"

app = Flask(__name__)

@app.route('/generate_token')
def generate_token():
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return jsonify({'error': 'Credentials file not found'}), 400
    
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')

    session['state'] = state
    return jsonify({'authorization_url': authorization_url})

@app.route('/oauth2callback')
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

@app.route('/token_saved')
def token_saved():
    return 'Token saved successfully.'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
