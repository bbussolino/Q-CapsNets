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

  - `model`: string with the name of the model to be used. Currently `ShallowCapsNet` [2] and `DeepCaps` [3] are supported. 
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
  ![equation](https://latex.codecogs.com/gif.latex?lr%20%3D%20lr_0%20%5Ccdot%20%5Cmbox%7Bdecay%5C_rate%7D%20%5E%20%7B%5Cfrac%7B%5Cmbox%7Btraining%5C_step%7D%7D%7B%5Cmbox%7Bdecay%5C_steps%7D%7D%7D)
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

The output of the framework, i.e., the resulting bitwidths and accuracy, are displayed on the terminal. The quantized model is stored in a .pt file. 

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
|ShallowCapsNet_mnist_top.pt|ShallowCapsNet|MNIST| 99.67% |
|ShallowCapsNet_fashionmnist_top.pt|ShallowCapsNet|FashionMNIST| 92.79% |
|DeepCaps_mnist_top.pt|DeepCaps|MNIST| 99.75% |
|DeepCaps_fashionmnist_top.pt|DeepCaps|FashionMNIST| 95.08% |
|DeepCaps_cifar10_top.pt|DeepCaps|CIFAR10| 91.26% |

Assuming you want to use the ShallowCapsNet weights for MNIST dataset and that they are stored in the folder "./pre_trained_weights/", you need to use the arguments: 
```sh 
--no-training --trained-weights-path ./pre_trained_weights/ShallowCapsNet_mnist_top.pt 
```
The output files of the framework will be stored in the same  "./pre_trained_weights/" folder and named ShallowCapsNet_mnist_top_quantized_xxx.pt, where xxx can stand for satisfied, memory or accuracy. 
### Arguments for usage with model training 
To control the training process, it is possible to set the parameters `epochs`, `lr`, `batch-size`, `regularization-scale`, `decay-steps`, `decay-rate`, and `hard-training`. The `full-precision-filename` argument is the directory where the full-precision model will be stored. A working example for the DeepCaps architecture with CIFAR10 dataset is the following: 
```sh 
--epochs 300 --lr 0.001 --batch-size 100 --regularization-scale 0.005 --decay-steps 6000 --decay-rate 0.96 --hard-training --full-precision-filename ./results/DeepCaps_cifar10.pt
```

The full precision model will be stored in the folder "./results/" named DeepCaps_cifar10.pt. The quantized model will be stored in the same folder and named DeepCaps_cifar10_quantized_xxx.pt, where xxx can stand for satisfied, memory or accuracy. 

### Arguments for Q-CapsNets framework 
To run the Q-CapsNets framework with 0.2% accuracy tolerance, 1MB memory budget and stochastic rounding it is necessary to used the arguments: 
```sh 
--accuracy-tolerance 0.2 --memory-budget 1 --quantization_method stochastic_rounding
```
The supported quantization methods are `round_to_nearest`, `stochastic_rounding`, `logarithmic`, and `truncation`. 

# License
MIT

# References 
[1] Marchisio, A., Bussolino, B., Colucci, A., Martina, M., Masera, G., & Shafique, M. (2020). Q-CapsNets: A Specialized Framework for Quantizing Capsule Networks. 2020 57th ACM/IEEE Design Automation Conference (DAC).

[2] Sabour, S., Frosst, N., & Hinton, G.E. (2017). Dynamic Routing Between Capsules. ArXiv, abs/1710.09829.

[3] Rajasegaran, J., Jayasundara, V., Jayasekara, S., Jayasekara, H., Seneviratne, S., & Rodrigo, R. (2019). DeepCaps: Going Deeper With Capsule Networks. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10717-10725.
