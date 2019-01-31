.. _api_guide_conv_en:

#############
Convolution
#############

Convolution has two sets of inputs: feature map and convolution kernel, depending on the input characteristics and the shape of the convolution kernel, the layout and the calculation method. In Fluid, there is a one-dimensional convolution for variable-length sequence features, two-dimensional (2D Conv) and three-dimensional convolution (3D Conv) for fixed-length image features. At the same time, there is also a reverse process of convolution calculation. The following describes the 2D/3D convolution in Fluid, and then introduces the sequence convolution.


2D/3D Convolution
==================

1. Input parameters of convolution:
--------------------------------------
The convolution needs to be determined according to stride, padding, filter size, groups, and dilation rate. Groups were first introduced in `AlexNet <https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf>`_ It can be considered that the original convolution is split into independent sets of convolution to be calculated.

  **Note**: In the same way as cuDNN, Fluid currently only supports filling feature map above and below with the same length , as well as left and right.

- Input and Output Layout: 

  Layout of input feature of 2D convolution is [N, C, H, W] or [N, H, W, C], N is batch size, C is passes, H,W is height and width of feature. Layout of input feature is the same as that of output feature.(Layout of input feature of 3D convolution is [N, C, D, H, W] or [N, D, H, W, C]. But **note**,Fluid convolution currently only supports [N, C, H, W],[N, C, D, H, W].)
   
- Layout of convolution kernel:
  
  The layout of the 2D convolutional convolution kernel (also called weight) in Fluid is [C_o, C_in / groups, f_h, f_w], C_o, C_in represent the number of output and input channels, and f_h and f_w represent the height and width of the convolution kernel window, which are stored in row order. (The corresponding 2D convolutional convolution kernel layout is [C_o, C_in / groups, f_d, f_h, d_w], which is also stored in row order.)
  
- Depthwise Separable Convolution: 
   
  Depthwise Separable Convolution contains depthwise convolution和pointwise convolution. The interfaces of these two convolutions are the same as the above conventional convolutional interfaces. The former can be done by setting groups for ordinary convolutions. The latter can be done by setting the size of the convolution kernel filters to 1x1. Depthwise Separable Convolution reduces the parameters as well as volume of computation.
  
  For depthwise convolution, you can set groups equal to the number of input channels. At this time, the convolution kernel shape of the 2D convolution is [C_o, 1, f_h, f_w]. For pointwise convolution, the shape of the convolution kernel is [C_o, C_in, 1, 1].
  
  **Note**: Fluid is highly optimized for GPU computing for depthwise convolution. You can use Fluid's own optimized CUDA program by setting :code:`use_cudnn=False` in the :code:`fluid.layers.conv2d` interface.
   
- Dilated Convolution:
  
  Compared with ordinary convolution, for dilated convolution, the convolution kernel is not continuous in the feature map, but is spaced. This interval is called dilation. When it is equal to 1, it is ordinary convolution. and dilated convolution is larger than ordinary convolution.
  

- API summary:
 - :ref:`api_fluid_layers_conv2d`
 - :ref:`api_fluid_layers_conv3d`
 - :ref:`api_fluid_layers_conv2d_transpose`
 - :ref:`api_fluid_layers_conv3d_transpose`


1D sequence convolution
=========================

Fluid can represent a variable-length sequence structure. The variable length here means that the number of time steps of different samples is different. It is usually represented by a 2D Tensor and an auxiliary structure that can distinguish the sample length. Assume that the shape of the 2D Tensor is shape, shape[0] is the total number of time steps for all samples, and shape[1] is the size of the sequence feature.

Convolution based on this data structure is called sequence convolution in Fluid and also represents one-dimensional convolution. Similiar to image convolution, the input parameters of the sequence convolution contain the size of convolution kernel, the size of fill, and the size of sliding step. But unlike the 2D convolution, the number of these parameters is 1. **Note**, it currently only supports stride is 1. The output sequence has the same number of time steps as the input sequence.

Supposing the input sequence shape is (T, N), while T is the number of time steps of the sequence, and N is the sequence feature size; The convolution kernel has a context step size of K. The output sequence length is M, the convolution kernel weight shape is (K * N, M), and the output sequence shape is (T, M).
  