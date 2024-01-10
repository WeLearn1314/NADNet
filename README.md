# NADNet
A lightweight and effective neuron attention CNN for image denoising (NADNet) by Jibin Deng*, Chaohua Hu is submitted to Signal, Image and Video Processing, 2024.
 
# Prerequisites:
python == 3.6.2

tensorflow == 2.0.0

keras == 2.3.1

opencv-python == 4.5.5.62

scikit-image == 0.17.2

# Prepare training dataset
Taking the gray training dataset "Train400" for example, please run:

python data_generator.py

# Denoising Training
For train the NADNet, please run:

python mainimprovement.py

# Denoising Testing
**The pretrained models have been uploaded to the folder "pretrained_models".**

For test the NADNet, please run:

python mainimprovement.py --pretrain sigma (e.g., 15, 25 and 50)/model_50.h5 --only_test True

# Denoising Datasets
The gray train dataset "Train400" you can download here (Selected in the paper):

https://www.dropbox.com/s/8j6b880m6ddxtee/TNRD-Codes.zip?dl=0&file_subpath=%2FTNRD-Codes%2FTrainingCodes4denoising%2FFoETrainingSets180

The color train dataset "BSD400" you can download here (Selected in the paper):

https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

The real-world train dataset "PolyU" you can download here (Selected in the paper):

https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset

The real-world test dataset "Nam" you can download here (Selected in the paper):

https://github.com/GuoShi28/CBDNet/tree/master/testsets/Nam_patches

The real-world test dataset "CC" you can download here (Selected in the paper):

https://github.com/csjunxu/MCWNNM_ICCV2017

# Deraining Datasets
The dataset "Rain100H" and "Rain100L" you can download here:

https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g