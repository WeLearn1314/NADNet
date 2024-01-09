# NADNet
A lightweight and effective neuron attention CNN for image denoising (NADNet) by Jibin Deng*, Chaohua Hu is submitted to Signal, Image and Video Processing, 2023.
 
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
For test the NADNet, please run:

python mainimprovement.py --pretrain sigma/model_50.h5 --only_test True

# Denoising Datasets
The gray train dataset "Train400" you can download here (Selected in the paper):

https://download.csdn.net/download/qq_41104871/87646484

The color train dataset "BSD400" you can download here (Selected in the paper):

https://download.csdn.net/download/qq_41104871/87647333

The real-world train dataset "PolyU" you can download here (Selected in the paper):

https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset

The real-world test dataset "Nam" you can download here (Selected in the paper):

https://github.com/GuoShi28/CBDNet/tree/master/testsets/Nam_patches

The real-world test dataset "CC" you can download here (Selected in the paper):

https://github.com/csjunxu/MCWNNM_ICCV2017

The real-world test dataset "DND" you can download here (no ground truth):

https://github.com/GuoShi28/CBDNet/tree/master/testsets/DND_patches

The real-world test dataset "SenseNoise" you can download here:

https://github.com/zhangyi-3/IDR | https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/Eqj2xo-jzTlChyuyF-JWmzQBdi5sLBrRZxQikdnko3EpfQ?e=81pFjp

# Deraining Datasets
The dataset "Rain100H" and "Rain100L" you can download here:

https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g