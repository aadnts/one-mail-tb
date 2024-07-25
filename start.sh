# Navigate to backend directory and run backend server
(
  cd /root/one-mail-tb/backend || exit
  source .venv/bin/activate
  uvicorn score:app --log-level debug --host 0.0.0.0 > /root/one-mail-tb/logs/backend.log 2>&1 &
  deactivate
) &

# Navigate to frontend directory and run frontend server
(
  cd /root/one-mail-tb/frontend || exit
  yarn run dev > /root/one-mail-tb/logs/frontend.log 2>&1 &
) &

# Navigate to gmail directory and run scripts
(
  cd /root/one-mail-tb/gmail || exit
  source .gmailenv/bin/activate
  python3 upload_file.py > /root/one-mail-tb/logs/upload_file.log 2>&1
  python3 cron.py > /root/one-mail-tb/logs/cron.log 2>&1
  deactivate
) &

echo "All specified commands have been executed in order."
