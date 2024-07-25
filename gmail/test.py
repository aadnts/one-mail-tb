from retrieve_emails import RetrieveEmail
from orchestrator import send_emails

r = RetrieveEmail()
r.retrieve_emails()
send_emails()