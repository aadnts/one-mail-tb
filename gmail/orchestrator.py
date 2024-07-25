from time import sleep
import requests
import os
from dotenv import load_dotenv
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

load_dotenv()

# Define the server URL
server_url = os.getenv("SERVER_URL")

# Define the folder, model, and chunk size
directory_path = "/root/one-mail-tb/gmail/threads"
chunk_size = 5 * 1024 * 1024  # 5 MB
model = "openai-gpt-4o-mini"
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE")

# Initialize Doctr OCR predictor
ocr_model = ocr_predictor(pretrained=True)

# Function to upload file in chunks
def upload_file_in_chunks(file_path, server_url, model, uri, username, password, database):
    total_chunks = os.path.getsize(file_path) // chunk_size + 1
    original_name = os.path.basename(file_path)
    
    with open(file_path, 'rb') as file:
        for chunk_number in range(1, total_chunks + 1):  # Ensure chunk numbering starts from 1
            chunk = file.read(chunk_size)
            files = {'file': (original_name, chunk)}
            data = {
                'chunkNumber': chunk_number,
                'totalChunks': total_chunks,
                'originalname': original_name,
                'model': model,
                'uri': uri,
                'userName': username,
                'password': password,
                'database': database
            }
            
            response = requests.post(f"{server_url}/upload", files=files, data=data)
            response_data = response.json()
            
            if response_data['status'] != 'Success':
                print(f"Error uploading chunk {chunk_number}: {response_data['message']}")
                return False
            else:
                print(f"Chunk {chunk_number}/{total_chunks} uploaded successfully.")
    
    return True

# Function to extract nodes and relations
def extract_nodes_and_relations(server_url, model, uri, username, password, database, file_name):
    data = {
        'uri': uri,
        'userName': username,
        'password': password,
        'database': database,
        'model': model,
        'file_name': file_name,
        'source_type': 'local file'
    }
    
    print(f"Sending extract request with data: {data}")
    response = requests.post(f"{server_url}/extract", data=data)
    print(f"Extraction endpoint response status: {response.status_code}")
    print(f"Extraction endpoint response content: {response.text}")
    
    try:
        response_data = response.json()
    except ValueError:
        print(f"Error parsing extraction response: {response.text}")
        return None
    
    return response_data

# Function to process and perform OCR on pdf or image
def process_pdf_or_image(file_path, predictor):
    try:
        doc = (
            DocumentFile.from_pdf(file_path)
            if file_path.endswith(".pdf")
            else DocumentFile.from_images(file_path)
        )
        result = predictor(doc)
        raw_export = result.render()
        if len(raw_export.strip()) < 20:
            return None
        return raw_export
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process files within threads
def process_files_in_thread(thread_path, ocr_model):
    for root, _, files in os.walk(thread_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.lower().endswith(('.jpg', '.png', '.pdf')):
                print(f"Processing OCR for file: {file_path}")
                ocr_text = process_pdf_or_image(file_path, ocr_model)
                if ocr_text:
                    txt_file_path = file_path.rsplit('.', 1)[0].replace(" ", "_") + ".txt"
                    with open(txt_file_path, "w") as txt_file:
                        txt_file.write(ocr_text)
                    os.remove(file_path)
                    print(f"Created {txt_file_path} and removed original file {file_path}")
                else:
                    print(f"No text extracted from {file_path}")

def send_emails():
    # Process all files in the directory recursively
    with open("/root/one-mail-tb/gmail/new_emails", "r") as fr:
        new_emails = set(fr.read().split("\n"))
    for file_id in os.listdir(directory_path):
        if file_id in new_emails:
            thread_path = os.path.join(directory_path, file_id)
            process_files_in_thread(thread_path, ocr_model)
            for file_name in os.listdir(thread_path):
                file_path = os.path.join(thread_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith(".txt"):
                    print(f"Uploading and processing file: {file_path}")
                    if upload_file_in_chunks(file_path, server_url, model, uri, username, password, database):
                        print("File uploaded successfully.")
                        
                        # Wait for a sufficient amount of time to ensure the server has time to merge the file
                        sleep(20)
                        
                        # Extract nodes and relations
                        extraction_response = extract_nodes_and_relations(server_url, model, uri, username, password, database, os.path.basename(file_path))
                        if extraction_response:
                            print("Extraction response:", extraction_response)
                        else:
                            print("Extraction failed.")
                    else:
                        print("File upload failed.")
                    
    with open("/root/one-mail-tb/gmail/new_emails", "w") as fr:
        fr.write("")

if __name__ == "__main__":
    send_emails()
