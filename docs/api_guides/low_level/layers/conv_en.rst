.. _api_guide_conv_en:

#############
Convolution
#############

Convolution has two sets of inputs: feature maps and convolution kernels. Depending on the input features, the shape of the convolution kernel, the layout and the calculation method, in Fluid, there is a one-dimensional convolution for variable-length sequence features, two-dimensional (2D Conv) and three-dimensional convolution (3D Conv) for fixed-length image features. At the same time, there is also a reverse(backward) process of convolution calculation. The subsequent content describes the 2D/3D convolution in Fluid, and then introduces the sequence convolution.


2D/3D Convolution
==================

1. Input parameters of convolution:
--------------------------------------
The convolution needs to be determined according to stride, padding, filter size, groups, and dilation rate. Groups were first introduced in `AlexNet <https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf>`_ . It can be considered that the original convolution is split into independent sets of convolution to be calculated.

**Note**: In the same way as cuDNN, Fluid currently only supports padding upper and lower part of feature maps with equal length , as well as that for left and right part.

- The layout(shape) of Input and Output :

  Layout of input feature of 2D convolution is [N, C, H, W] or [N, H, W, C], where N is the batch size, C is the number of channels, H,W is the height and width of feature. Layout of input feature is the same as that of output feature. (Layout of input feature of 3D convolution is [N, C, D, H, W] or [N, D, H, W, C]. But **note**, Fluid convolution currently only supports [N, C, H, W],[N, C, D, H, W].)

- The layout of convolution kernel:

  The layout of the 2D_conv convolution kernel (also called weight) in Fluid is [C_o, C_in / groups, f_h, f_w], where C_o, C_in represent the number of output and input channels, and f_h, f_w represent the height and width of filter, which are stored in row order. (The corresponding 2D_conv convolution kernel layout is [C_o, C_in / groups, f_d, f_h, d_w], which is also stored in row order.)

- Depthwise Separable Convolution:

  Depthwise Separable Convolution contains depthwise convolution 和 pointwise convolution. The interfaces of these two convolutions are the same as the above normal convolutional interfaces. The former can be performed by setting groups for ordinary convolutions. The latter can be realised by setting the size of the convolution kernel filters to 1x1. Depthwise Separable Convolution reduces the parameters as well as the volume of computation.

  For depthwise convolution, you can set groups equal to the number of input channels. At this time, the convolution kernel shape of the 2D convolution is [C_o, 1, f_h, f_w]. For pointwise convolution, the shape of the convolution kernel is [C_o, C_in, 1, 1].

  **Note**: Fluid optimized GPU computing for depthwise convolution greatly. You can use Fluid's self-optimized CUDA program by setting :code:`use_cudnn=False` in the :code:`fluid.layers.conv2d` interface.

- Dilated Convolution:

  Compared with ordinary convolution, for dilated convolution, the convolution kernel does not continuously read values from the feature map, but with intervals. This interval is called dilation. When it is equal to 1, it becomes ordinary convolution. And receptive fields of dilated convolution is larger than that of ordinary convolution.


- related API:
 - :ref:`api_fluid_layers_conv2d`
 - :ref:`api_fluid_layers_conv3d`
 - :ref:`api_fluid_layers_conv2d_transpose`
 - :ref:`api_fluid_layers_conv3d_transpose`


1D sequence convolution
=========================

Fluid can represent a variable-length sequence structure. The variable length here means that the number of time steps of different samples is different. It is usually represented by a 2D Tensor and an auxiliary structure that can distinguish the sample length. Assume that the shape of the 2D Tensor is shape, shape[0] is the total number of time steps for all samples, and shape[1] is the size of the sequence feature.

Convolution based on this data structure is called sequence convolution in Fluid and also represents one-dimensional convolution. Similar to image convolution, the input parameters of the sequence convolution contain the filter size, the padding size, and the size of sliding stride. But unlike the 2D convolution, the number of each parameter is 1. **Note**, it currently only supports stride = 1. The output sequence has the same number of time steps as the input sequence.

Suppose the input sequence shape is (T, N), while T is the number of time steps of the sequence, and N is the sequence feature size; The convolution kernel has a context stride of K. The length of output sequence is M, the shape of convolution kernel weight is (K * N, M), and the shape of output sequence is (T, M).

- related API:
 - :ref:`api_fluid_layers_sequence_conv`
 - :ref:`api_fluid_layers_row_conv`
