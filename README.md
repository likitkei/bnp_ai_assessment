# # BNP Paribas AI Assessment Task

Task: Create an English word image generator and then feed the images to a 		  	machine learning model to recognize the word from the image.

Bonus : Push the program as docker image to Docker hub

## Development Environment
Programming Language: Python 3.6.6
Operating System: Ubuntu 18.04

## How To Run?
```sh
$ pip install requirements.txt
```
```sh
$ python run.py
```
The program will run and generate 100 text images (which will take few 5 - 10 seconds), and then test images will be fed to a convolutional recurrent neural network with a pre-trained model (Transfer learning) for recognition. Results will be returned and displayed in the terminal.

The generated text images, true_words.txt, and prediction_words.txt are stored in ./data folder.

## Notes
As stated in the requirement.txt, pytorch-cpu will be installed by default. For better performance, pytorch with GPU support could be installed and the program will utilize CUDA processors for computation.

## Reference
[CRNN in Pytorch](https://github.com/meijieru/crnn.pytorch)

[Text Image Generator](https://github.com/Belval/TextRecognitionDataGenerator)

[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717) 
