import { DataComponentProps } from '../../../types';
import OCRlogo from '../../../assets/images/OCR-logo.png';
import CustomButton from '../../UI/CustomButton';
import { buttonCaptions } from '../../../utils/Constants';

const OCRComponent: React.FC<DataComponentProps> = ({ openModal }) => {
  return (
    <CustomButton title={buttonCaptions.OCR} openModal={openModal} logo={OCRlogo} wrapperclassName='' className='' />
  );
};

export default OCRComponent;
