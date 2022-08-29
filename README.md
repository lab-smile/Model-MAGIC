# Model MAGIC: Diagnostically competitive performance of a physiology-informed generative adversarial network for contrast-free CT perfusion map generation

MAGIC is a novel, multitask deep network architecture that enables translation from noncontrast-enhanced CT imaging to CT perfusion imaging. This framework enables the contrast-free synthesis of perfusion imaging of the brain, and it is generalizable to other modalities of imaging as well.

The novelties of this framework for the contrast-free synthesis of perfusion imaging, in comparison to other modern image-to-image architectures, are as follows: 
- No existing framework to enable contrast-free perfusion imaging
- Physiologically-informed loss terms
- Multitask, simultaneous generation of four CTP maps 
- Shared encoding layers between perfusion maps
- Real-time, rapid generation of perfusion maps
- Improved spatial resolution of perfusion maps in the axial direction
- Generalizable to any additional perfusion maps
- Physicians-in-the-Loop Module

## System Requirements

We recommend using the Linux operating system. All listed commands in this part are based on the Linux operation system. We highly recommend using a GPU system for computation, but these code and directions are compatible with CPU only. 

We used Linux (GNU/Linux 3.10.0-1062.18.1.el7.x86_64) and an NVIDIA TITAN X GPU with CUDA version 7.6.5. 
### Prerequisites
The main dependent packages for Model MAGIC are as follows:
- Matlab version R2021B
- Python version 3.6.10
- PyTorch version 1.5.0
- OpenCV version 4.2.0.34
- NumPy version 1.16.0
- Scikit-learn version 0.22.1
- Sewar version 0.4.5
- CUDA version 7.6.5

Please refer the other dependent packages for Model MAGIC [here](magic_env.yml).

## Usage
### Running Time
- Installation time: The time for environmental setting up is about 40-50 minutes. 
- The image-to-map prediction speed is approximately 46 ms/ slice, which is about 7.43 sec per volume of 161 slices once trained.

### Sample training dataset
We provide a small sample training set for evaluation and introduction to this project's code. This can be found in [training_data](sample_train/). Contained in this dataset are one subfolders, for training a model. The training set contains 48 samples. The original MAGIC model was trained on over 16,000+ individual samples, but this sample set illustrates the program's functionality.

### Training network
Once the environment and training data are ready, you can directly change the directory in the [training_script](code/magic_train.py) and training the model by using the following command:
```
python magic_train.py 
```
You can specify learning rates, output save direction, number of training epochs, etc. using command line arguments. For example:  
```
python magic_train.py --dataset '../sample' --lrG 0.00005 --lrD 0.00005 --train_epoch 50 --save_root 'results' 
```
Or you can also training the model by using the [dockerfile](docker/training) we provided. Please find the dockerfile tutorial [here](docker/)

### Testing 
We provide four [testing_images](test/) and a pre-trained model (Please find our pre-trained model by using this google drive link: https://drive.google.com/file/d/1V-Cc2cBUbp9RL09unxhjfNbgVL9K1h1L/view?usp=sharing) for testing process.

You need to determine the ```dataset``` and ```save_root``` arguments before testing progress. 
You can directly change the directory in the [testing_script](code/magic_test.py) and testing the model by using the following command:
```
python magic_test.py --dataset '../sample' --save_root 'results' 
```

The expected results of these four test images are shown below:
![](https://github.com/lab-smile/Model-MAGIC/blob/main/img/expected_results.png)

## Citation

## Acknowledgement
The code was modified from https://github.com/mrzhu-cool/pix2pix-pytorch

## Contact
Any discussion, suggestions and questions please contact:
[Garrett Fullerton](mailto:gfullerton245@gmail.com), [Dr. Ruogu Fang](mailto:ruogu.fang@bme.ufl.edu).
Smart Medical Informatics Learning & Evaluation Laboratory, Dept. of Biomedical Engineering, University of Florida
