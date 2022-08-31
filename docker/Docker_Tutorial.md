## Training and Testing Dockerfile Tutorial Windows 10 version

### Intro
Docker is a containerization technology that supports the creation and use of Linux containers. Docker provides an image-based deployment model which makes it can share an application or service group with its dependencies across multiple environments.

Simply put, Docker can makes user deliver software quickly. Instead of running different commands across different environments. Users can simply re-build and reproduce code by running the dockerfile in the image.

The purpose of this manual is to help you reproduce the MAGIC train and test code through the dockerfile we provided.  

### Prerequistes
You need to download and install Docker on your platform by the following link: https://docs.docker.com/get-docker/

For windows system, you need to ensure *the Use WSL 2 instead of Hyper-V* when prompted.

Then you can double-click to open the **Docker Desktop**. When you see the small whale in the lower left corner turns green, the docker is ready! 

### Build docker 
Firstly you need to download our docker folder which includes the dockerfile and needed files. We used Linux and GPU with CUDA in this code, to prepare the environment for the docker, You need to run the following command to pull an image for the whole project:
```
docker pull nvidia/cuda:11.7.1-devel-ubuntu20.04
```
Then you can build a new container image to run your dockerfile. To do this, you can open a terminal and go the the directory with the dockerfile. Now build the container image using the ```docker build``` command:
```
docker build -t magic:1 .
```

Please save the docker folder under your base directory so you don't have to modify the dockerfile. For example, the directory for the docker folder in the local test computer is ```C:\test_docker\```.


### Docker for training
The training docker subfolder includes a zip file and the docker file. You don't have to unzip, we have the unzip command in our docker file. You need to move the training dataset under the same subfolder with the training code. Then you can simply run the following results to obtain the training results:
```
docker run -it magic:1
```
Dockerfile will automatically build the code environment, run the training script and save the results.

### Docker for testing
The testing docker subfolder includes all needed files for the testing part and the dockerfile. We also provide four sample test image [here](/docker/test_docker/test). If you want to obtained the same results as our expected test results, you need to download and save our pre-trained model under the same subfolder with the dockerfile by this link: https://drive.google.com/file/d/1V-Cc2cBUbp9RL09unxhjfNbgVL9K1h1L/view?usp=sharing 

Then you can simply run the following results to obtain the testing expected results:
```
docker run -it --rm --gpus=all -v /c/test_docker/run.sh:/run.sh -v /c/test_docker:/app -v /c/test_docker/results:/app_results/test_results magic:1 
```
This command also projected the test results from the image to your local computer. You will find the result in a subfolder titled *test_folder/results* which included four expected results.

If you are not used to using docker, you can also reproduce the results by following our README file.