import { Typography } from '@neo4j-ndl/react'; // Assurez-vous d'importer le composant Button
import React, { useState } from 'react';
import CustomModal from '../../../HOC/CustomModal';
import { S3ModalProps } from '../../../types';
import { buttonCaptions } from '../../../utils/Constants';

const OCRModal: React.FC<S3ModalProps> = ({ hideModal, open }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null); // État pour gérer le fichier sélectionné
  const [status, setStatus] = useState<'unknown' | 'success' | 'info' | 'warning' | 'danger'>('unknown');
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
  };

  const submitHandler = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      setIsLoading(true);

      try {
        const response = await fetch('http://104.248.236.94:5000/ocr', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          setStatus('success');
          setStatusMessage(`Fichier ${selectedFile.name} chargé avec succès`);
        } else {
          setStatus('danger');
          setStatusMessage('Erreur lors du chargement du fichier');
        }
      } catch (error) {
        console.error('Erreur:', error);
        setStatus('danger');
        setStatusMessage('Erreur lors du chargement du fichier');
      } finally {
        setIsLoading(false); // Fin du chargement
      }
    } else {
      setStatus('warning');
      setStatusMessage('Veuillez sélectionner un fichier avant de soumettre');
    }

    setTimeout(() => {
      hideModal();
      setStatus('unknown');
      setStatusMessage('');
      setSelectedFile(null);
    }, 3000);
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
      submitHandler={submitHandler}
      status={status}
      setStatus={setStatus}
      submitLabel={buttonCaptions.submit}
      isLoading={isLoading}
    >
      <div className='w-full inline-block'>
        <Typography variant='h4' className='py-5'>Upload your OCR credentials</Typography>
        <input type='file' onChange={handleFileChange} />
      </div>
    </CustomModal>
  );
};

export default OCRModal;
