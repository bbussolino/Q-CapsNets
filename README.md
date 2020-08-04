# Q-CapsNets framework 
### A Specialized Framework for Quantizing Capsule Networks 
Q-CapsNets [1] is a framework for the automated quantization of Capsule Networks presented at DAC2020. Please refer to the paper for the implementation details. 

# Environment 
The framework has been developed in an environment supporting: 

  - Python 3.6.4
  - PyTorch 1.0.1
  - Cudatoolkit 10.0.130
  - Cudnn 7.3.1
 
For Anaconda users, a working environment can be created and activated through the provided env_bussolino.yml file, running the commands: 
```sh
$ conda env create -f env_bussolino.yml
$ conda activate env_bussolino
```

# Arguments of the code 
Here the arguments of the framework are listed. For the defaul values, run: 
```sh
$ python main.py -h
```

  - `model`: string with the name of the model to be used. Currently `ShallowCapsNet` and `DeepCaps` are supported. 
  - `model-args`: string with the parameters of the model. The values are `[input_wh, input_ch, num_classes, dim_otuput_capsules]`, that are the width/height of the input images in pixels, the number of channels of the input images, the number of classes of the dataset and the dimension of the capsules of the last layer. 
  - `decoder`: name of the decoder to be used. Currently `FCDecoder`, `ConvDecoder28` and `ConvDecoder64` are supported. 
  - `decoder-args`: string with the parameters of the decoder. The values are `[in_dim, out_dim]` for the FCDecoder, that are the dimension of the input capsule and the dimension of the output image in pixel*pixel. For the ConvDecoders, the values are `[input_size, out_channels]`, that are the dimension of the input capsule and the number of channels of the output image. 
  - `dataset`: name of the dataset. Currently `mnist`, `fashion-mnist`, `cifar10`, and `svhn` are supported. 
  - `no-training`: toggle if you want to skip the training and use pre-trained weights 
  - `full-precision-filename`: string with the directory in which the full-precision-trained model will be stored. 
  - `trained-weights-path`: string with the directory of the pre-trained weights. 
  - `epochs`: number of epochs for the training 
  - `lr`: initial value of the learning rate 
  - `batch-size`: batch-size used for the training 
  - `log-interval`: during the training, information on losses and accuracy are displayed every log-interval steps. 
  - `regularization-scale`: multiplication factor to scale down the regularization loss of the decoder. 
  - `decay-steps`, `decay-rate`: the learning rate is decreased exponentially according to the formula below and these two parameters control the decayment. 
  ![equation](http://www.sciweavers.org/tex2img.php?eq=lr%20%3D%20lr_0%20%5Ccdot%20%5Cmbox%7Bdecay%5C_rate%7D%5E%7B%5Cfrac%7B%5Cmbox%7Btraining%5C_step%7D%7D%7B%5Cmbox%7Bdecay%5C_steps%7D%7D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
  - `hard-training`: toggles hard training after half of the training epochs. 
  - `test-batch-size`: batch size used during testing 
  - `visible-gpus`: set the number of the GPU to be used. Set to '-1' to use the CPU. 
  - `threads`: set the number of threads for the data loader. 
  - `seed`: set the GPU seed for weights initialization. 
  - `accuracy-tolerance`: accuracy tolerance for the Q-CapsNets framework 
  - `quantization_method`: quantization method for the Q-CapsNets framework 
  - `memory-budget`: memory budget for the Q-CapsNets framework 

# Usage 
Q-CapsNets can be run by command line setting all the necessary arguments. A first set of arguments is necessary for the model instantiation. Then the second set of arguments changes depending on whether pre-trained weights are available or the model must be trained. The last set of arguments is related to the Q-CapsNets framework. To run the code run the command 

```sh 
$ python main.py 
```

followed by the needed arguments, that are explained in the following subsections. 

### Arguments for model setting 
ShallowCapsNet for MNIST: 
```sh 
--model ShallowCapsNet --model-args 28 1 10 16 --decoder FCDecoder --decoder-args 16 784 --dataset mnist 
```
ShallowCapsNet for FashionMNIST: 
```sh 
--model ShallowCapsNet --model-args 28 1 10 16 --decoder FCDecoder --decoder-args 16 784 --dataset fashion-mnist
``` 
ShallowCapsNet for CIFAR10: 
```sh 
--model ShallowCapsNet --model-args 64 3 10 16 --decoder FCDecoder --decoder-args 16 4096 --dataset cifar10
``` 
DeepCaps for MNIST: 
```sh 
--model DeepCaps --model-args 28 1 10 32 --decoder ConvDecoder28 --decoder-args 32 1 --dataset mnist 
```
DeepCaps for FashionMNIST: 
```sh 
--model DeepCaps --model-args 28 1 10 32 --decoder ConvDecoder28 --decoder-args 32 1 --dataset fashion-mnist
```
DeepCaps for CIFAR10: 
```sh 
--model DeepCaps --model-args 64 3 10 32 -decoder ConvDecoder64 --decoder-args 32 3 --dataset cifar10
```
DeepCaps for SVHN: 
```sh 
--model DeepCaps --model-args 64 3 10 32 -decoder ConvDecoder64 --decoder-args 32 3 --dataset svhn
```
### Arguments for usage with pre-trained weights 
You can download the pre-trained weights at the following link: 
https://drive.google.com/file/d/1vmQzlrD3O0r_vyogzxoGBf1MKYfwcU9L/view?usp=sharing

The available weights are listed in the table below. 
| Filename | Model | Dataset | Accuracy |
|----------|-------|---------|----------|
|ShallowCapsNet_mnist_top.pt|ShallowCapsNet|MNIST|  |
|ShallowCapsNet_fashionmnist_top.pt|ShallowCapsNet|FashionMNIST|  |
|DeepCaps_mnist_top.pt|DeepCaps|MNIST|
|DeepCaps_fashionmnist_top.pt|DeepCaps|FashionMNIST|
|DeepCaps_cifar10_top.pt|DeepCaps|CIFAR10|

Assuming you want to use the ShallowCapsNet weights for MNIST dataset and that they are stored in the folder "./pre_trained_weights/", you need to use the arguments: 
```sh 
-- no-training --trained-weights-path ./pre_trained_weights/ShallowCapsNet_mnist_top.pt 
```
### Arguments for usage with model training 



# License
MIT

# References 
[1]
[2]
[3]
