import os
import base64
import json
import pickle
from datetime import datetime
import string
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
import re

class RetrieveEmail:
    # Constants
    JSON_FILE_PATH = "threads_metadata.json"
    THREADS_FOLDER_PATH = "threads"

    # Load credentials from the token file
    def __init__(self) -> None:   
        creds = None
        if os.path.exists("/root/one-mail-tb/gmail/credentials/token.pickle"):
            with open("/root/one-mail-tb/gmail/credentials/token.pickle", "rb") as token:
                creds = pickle.load(token)

        # If there are no valid credentials, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise ValueError("No valid credentials provided.")

        # Build the Gmail API service
        self.service = build("gmail", "v1", credentials=creds)

    def load_threads_metadata(self):
        if os.path.exists(self.JSON_FILE_PATH):
            with open(self.JSON_FILE_PATH, "r") as f:
                return json.load(f)
        return {}

    def save_threads_metadata(self, threads_metadata):
        with open(self.JSON_FILE_PATH, "w") as f:
            json.dump(threads_metadata, f, indent=4)

    def get_threads(self, user_id="me", label_ids=[], max_results=10):
        try:
            response = (
                self.service.users()
                .threads()
                .list(userId=user_id, labelIds=label_ids, maxResults=max_results)
                .execute()
            )
            threads = response.get("threads", [])
            return threads
        except HttpError as error:
            print(f"An error occurred: {error}")
            return None

    def create_email_folder(self, thread_folder_path, messageId):
        email_folder_path = os.path.join(thread_folder_path, messageId)
        if not os.path.exists(email_folder_path):
             os.makedirs(email_folder_path)
        return email_folder_path
        
    # def create_email_folder(self, thread_folder_path, from_email, internal_date):
    #     date_str = internal_date.split(" ")[0]
    #     time_str = internal_date.split(" ")[1]
    #     email_folder_name = f"{from_email}:{date_str}--{time_str.replace(':', '-')}"
    #     email_folder_path = os.path.join(thread_folder_path, email_folder_name)
    #     if not os.path.exists(email_folder_path):
    #         os.makedirs(email_folder_path)
    #     return email_folder_path

    def get_thread_details(self, thread_id, threads_metadata, user_id="me"):
        try:
            thread = self.service.users().threads().get(userId=user_id, id=thread_id).execute()
            messages = thread.get("messages", [])

            for message in messages:
                message_id = message["id"]

                # Check if message_id is already in the mail_ids list
                if message_id not in threads_metadata[thread_id].get("mail_ids", []):
                    # Process the message
                    success = self.process_message(message, thread_id, threads_metadata)

                    if success:
                        # Add the message_id to the mail_ids list
                        threads_metadata[thread_id]["mail_ids"].append(message_id)
                        with open("new_emails", "a") as f:
                            f.write("\n" + message["id"])

        except HttpError as error:
            print(f"An error occurred: {error}")

    def extract_sender_email(self, sender_info):
        # Extract the email address from the 'From' field
        return sender_info.split()[-1].strip('<>')

    def extract_subject(self, subject_info):
        # Extract the subject line
        return subject_info.replace(' ', '_')  # Replace spaces with underscores for filenames

    def process_message(self, message, thread_id, threads_metadata):
        try:
            headers = {
                header["name"]: header["value"] for header in message["payload"]["headers"]
            }
            from_email = self.extract_sender_email(headers.get("From", ""))
            subject = self.extract_subject(headers.get("Subject", ""))
            subject = subject.translate(str.maketrans('', '', string.punctuation))
            subject = subject.replace(' ', '_')
            
            # Access labelIds directly from message
            labels = message.get("labelIds", [])
            
            # Check if "CATEGORY_PERSONAL" is in the labels
            if "CATEGORY_PERSONAL" not in labels:
                print(f"Skipping email {message['id']} as it does not have CATEGORY_PERSONAL label.")
                return  # Skip this email

            internal_date = datetime.fromtimestamp(
                int(message.get("internalDate")) / 1000
            ).strftime("%Y-%m-%d %H:%M:%S")

            # Extract the email content
            email_message = self.extract_latest_text(message["payload"])

            # Remove previous conversations
            email_message = self.remove_previous_conversations(email_message)

            # Create or update the thread folder
            thread_folder_path = self.THREADS_FOLDER_PATH

            # Create a folder for the email inside the thread folder
            email_folder_path = self.create_email_folder(
                thread_folder_path, message["id"] #from_email, internal_date
            )

            # Extract metadata
            metadata = {
                "id": message["id"],
                "snippet": message.get("snippet"),
                "historyId": message.get("historyId"),
                "internalDate": internal_date,
                "sizeEstimate": message.get("sizeEstimate"),
                "threadId": message.get("threadId"),
                "labelIds": labels,
                "headers": headers,
            }

            # Save email text to a file
            email_text_file = os.path.join(email_folder_path, f"{from_email}_{subject}.txt")
            with open(email_text_file, "w") as f:
                f.write(email_message)

            # Save metadata to a JSON file
            metadata_file = os.path.join(email_folder_path, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)

            # Download and save attachments
            self.get_attachments(message, email_folder_path)
            
            return True
            
            # Update threads metadata with the email ID
            # if message["id"] not in threads_metadata[thread_id]["mail_ids"]:
            #     with open("new_emails", "a") as f:
            #         f.write("\n" + message["id"])
            #     threads_metadata[thread_id]["mail_ids"].append(message["id"])

        except HttpError as error:
            print(f"An error occurred: {error}")
            return False

    def extract_latest_text(self, payload):
        """Extracts the latest email text content."""
        if "parts" in payload:
            for part in payload["parts"]:
                if (
                    part["mimeType"] == "text/plain"
                    and "body" in part
                    and "data" in part["body"]
                ):
                    msg_str = base64.urlsafe_b64decode(part["body"]["data"].encode("ASCII"))
                    return msg_str.decode("utf-8")
                elif "parts" in part:
                    return self.extract_latest_text(part)
        elif "body" in payload and "data" in payload["body"]:
            msg_str = base64.urlsafe_b64decode(payload["body"]["data"].encode("ASCII"))
            return msg_str.decode("utf-8")
        return ""

    def remove_previous_conversations(self, email_message):
        """Removes previous conversations from the email message."""
        # Regular expression to detect lines indicating previous email content
        # previous_email_pattern_fr = re.compile(r"(-{2,}|De :|Envoyé :|À :|Objet :)")

        lines = email_message.split("\n")
        new_lines = []

        for line in lines:
            # if previous_email_pattern_fr.match(line):
            #     break
            new_lines.append(line)

        return "\n".join(new_lines)

    def get_attachments(self, message, folder_name):
        if "parts" in message["payload"]:
            for part in message["payload"]["parts"]:
                if part["filename"]:
                    attachment_id = part["body"]["attachmentId"]
                    attachment = (
                        self.service.users()
                        .messages()
                        .attachments()
                        .get(userId="me", messageId=message["id"], id=attachment_id)
                        .execute()
                    )
                    data = base64.urlsafe_b64decode(attachment["data"].encode("UTF-8"))
                    path = os.path.join(folder_name, part["filename"])
                    with open(path, "wb") as f:
                        f.write(data)
                        print(f'Attachment {part["filename"]} downloaded.')

    def retrieve_emails(self):
        threads_metadata = self.load_threads_metadata()
        threads = self.get_threads(max_results=60)  # Retrieve the 30 latest threads

        if threads:
            for thread in threads:
                thread_id = thread["id"]
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if thread_id not in threads_metadata:
                    # New thread, create a new entry in the JSON file
                    threads_metadata[thread_id] = {
                        "id": thread_id,
                        "mail_ids": [],
                        "created_at": current_datetime,
                    }
                else:
                    # Existing thread, update the "created_at" value with the current time
                    threads_metadata[thread_id]["created_at"] = current_datetime

                # Process thread details only if it's not already processed
                self.get_thread_details(thread_id, threads_metadata)

        self.save_threads_metadata(threads_metadata)

if __name__ == "__main__":
    email_retriever = RetrieveEmail()
    email_retriever.retrieve_emails()
