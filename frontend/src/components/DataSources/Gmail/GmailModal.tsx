import React, { useState } from 'react';
import CustomModal from '../../../HOC/CustomModal';
import { S3ModalProps } from '../../../types';
import { buttonCaptions } from '../../../utils/Constants';
import { Typography } from '@neo4j-ndl/react';

const GmailModal: React.FC<S3ModalProps> = ({ hideModal, open }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'unknown' | 'success' | 'info' | 'warning' | 'danger'>('unknown');
  const [statusMessage, setStatusMessage] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
  };

  const uploadFile = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch('http://104.248.236.94:5000/upload_credentials', {
          method: 'POST',
          body: formData,
        });

        const responseBody = await response.json();

        console.log('Response status:', response.status);
        console.log('Response body:', responseBody);

        if (response.ok) {
          console.log('File uploaded successfully');
          authenticateWithGoogle();
        } else {
          console.error('Error uploading file:', responseBody);
          setStatus('danger');
          setStatusMessage(`Error uploading file: ${responseBody.error}`);
        }
      } catch (error) {
        console.error('Error:', error);
        setStatus('danger');
        setStatusMessage('Error uploading file');
      }
    } else {
      setStatus('warning');
      setStatusMessage('Please select a file before submitting');
    }
  };

  const authenticateWithGoogle = async () => {
    try {
      const response = await fetch('http://104.248.236.94:5000/generate_token');
      const data = await response.json();
      window.location.href = data.authorization_url;
    } catch (error) {
      console.error('Error during authentication:', error);
      setStatus('danger');
      setStatusMessage('Error during authentication');
    }
  };

  const onClose = () => {
    hideModal();
    setSelectedFile(null);
    setStatus('unknown');
    setStatusMessage('');
  };

  return (
    <CustomModal
      open={open}
      onClose={onClose}
      statusMessage={statusMessage}
      submitHandler={uploadFile}
      status={status}
      setStatus={setStatus}
      submitLabel={buttonCaptions.submit}
    >
      <div className='w-full inline-block'>
        <Typography variant='h4' className='py-5'>Upload your gmail credentials</Typography>
        <input type='file' onChange={handleFileChange} />
      </div>
    </CustomModal>
  );
};

export default GmailModal;
