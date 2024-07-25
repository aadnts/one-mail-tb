import React from 'react';
import CustomButton from '../../UI/CustomButton';
import { buttonCaptions } from '../../../utils/Constants';
import gmaillogo from '../../../assets/images/gmail-logo.png';
import { DataComponentProps } from '../../../types';

const GmailComponent: React.FC<DataComponentProps> = ({ openModal }) => {
  return (
    <CustomButton title={buttonCaptions.gmail} openModal={openModal} logo={gmaillogo} wrapperclassName='' className='' />
  );
};

export default GmailComponent;
