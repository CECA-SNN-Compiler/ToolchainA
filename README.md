# ANN to SNN tool

This is a tool for transferring ANN to SNN.
It supports low-precision quantization of network weights.
The tool supports the networks with convolution layer,
linear layer and average pooling layer.

## Requirements
* Python >=3.5
* pytorch >=1.3
* GPUtil
* matplotlib

## Constraints
For Hardware v1.0, the input channel size of the convolution
in the network
must meet the following constraints.
* 3x3 kernel
  * 1bit in_channels<=113
  * 2bit in_channels<=56
  * 3bit in_channels<=37
  * 4bit in_channels<=28
* 5x5 kernel
  * 1bit in_channels<=40
  * 2bit in_channels<=20
  * 3bit in_channels<=13
  * 4bit in_channels<=10

The convolutional layers in the ANN must
take the ReLU activations function after it.

The final Fully connect layer and softmax layer will
not be transformed.

## Transform process

![process_pic](./pics/process.svg)

There is a example at transform_example.py.
You can transform your own network by modifying it.
The arguments here are:
* --load_file LOAD_FILE
                        the location of the trained weights
*  --dataset DATASET     the location of the trained weights
*  --save_file SAVE_FILE
                        the output location of the transferred weights
*   --batch_size BATCH_SIZE
*   --test_batch_size TEST_BATCH_SIZE
*   --timesteps TIMESTEPS
*   --reset_mode {zero,subtraction}
*   --weight_bitwidth WEIGHT_BITWIDTH
                        weight quantization bitwidth
*   --finetune_lr FINETUNE_LR
                        finetune learning rate
*   --finetune_epochs FINETUNE_EPOCHS
                        finetune epochs
*   --finetune_wd FINETUNE_WD
                        finetune weight decay
*   --finetune_momentum FINETUNE_MOMENTUM
                        finetune momentum


### dataset
The datasets.py contains the dataloader for MNIST and CIFAR10.

If other dataset is used,
the corresponding dataloader should be defined in the datasets.py

### Write network definition
Write your network class use the following modules:
* SpikeReLU
* SpikeConv2d,SpikeLinear
* SpikeAvgPool2d,spike_avg_pooling

###