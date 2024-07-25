import os
import time
from retrieve_emails import RetrieveEmail
from orchestrator import send_emails

def job():
    r = RetrieveEmail()
    r.retrieve_emails()
    send_emails()

if __name__ == "__main__":
    CREDENTIALS_FILE_PATH = "/root/one-mail-tb/gmail/credentials/token.pickle"
    
    while True:
        if os.path.exists(CREDENTIALS_FILE_PATH):
            job()
        else:
            print(f"'{CREDENTIALS_FILE_PATH}' not found. Skipping job execution.")
        
        time.sleep(60)  # Wait for 60 seconds before running the job again