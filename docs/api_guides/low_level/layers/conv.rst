.. _api_guide_conv:

#####
卷积
#####

卷积有两组输入：特征图和卷积核，依据输入特征和卷积核的形状、Layout不同、计算方式的不同，在Fluid里，有针对变长序列特征的一维卷积，有针对定长图像特征的二维(2D Conv)、三维卷积(3D Conv)，同时也有卷积计算的逆向过程，下面先介绍Fluid里的2D/3D卷积，再来介绍序列卷积。


2D/3D卷积
==============

1. 卷积输入参数：
---------------------

卷积需要依据滑动步长(stride)、填充长度(padding)、卷积核窗口大小(filter size)、分组数(groups)、扩张系数(dilation rate)来决定如何计算。groups最早在 `AlexNet <https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf>`_ 中引入, 可以理解为将原始的卷积分为独立若干组卷积计算。

  **注意**: 同cuDNN的方式，Fluid目前只支持在特征图上下填充相同的长度，左右也是。

- 输入输出Layout:

  2D卷积输入特征的Layout为[N, C, H, W]或[N, H, W, C], N即batch size，C是通道数，H、W是特征的高度和宽度，输出特征和输入特征的Layout一致。(相应的3D卷积输入特征的Layout为[N, C, D, H, W]或[N, D, H, W, C]，但 **注意**，Fluid的卷积当前只支持[N, C, H, W]，[N, C, D, H, W]。)

- 卷积核的Layout:

  Fluid中2D卷积的卷积核(也称权重)的Layout为[C_o, C_in / groups, f_h, f_w]，C_o、C_in表示输出、输入通道数，f_h、f_w表示卷积核窗口的高度和宽度，按行序存储。(相应的3D卷积的卷积核Layout为[C_o, C_in / groups, f_d, f_h, d_w]，同样按行序存储。)

- 深度可分离卷积(depthwise separable convolution):

  在深度可分离卷积中包括depthwise convolution和pointwise convolution两组，这两个卷积的接口和上述普通卷积接口相同。前者可以通过给普通卷积设置groups来做，后者通过设置卷积核filters的大小为1x1，深度可分离卷积减少参数的同时减少了计算量。

  对于depthwise convolution，可以设置groups等于输入通道数，此时，2D卷积的卷积核形状为[C_o, 1, f_h, f_w]。
  对于pointwise convolution，卷积核的形状为[C_o, C_in, 1, 1]。

  **注意**：Fluid针对depthwise convolution的GPU计算做了高度优化，您可以通过在
  :code:`fluid.layers.conv2d` 接口设置 :code:`use_cudnn=False` 来使用Fluid自身优化的CUDA程序。

- 空洞卷积(dilated convolution):

  空洞卷积相比普通卷积而言，卷积核在特征图上取值时不在连续，而是间隔的，这个间隔数称作dilation，等于1时，即为普通卷积，空洞卷积相比普通卷积的感受野更大。

- API汇总:
 - :ref:`cn_api_fluid_layers_conv2d`
 - :ref:`cn_api_fluid_layers_conv3d`
 - :ref:`cn_api_fluid_layers_conv2d_transpose`
 - :ref:`cn_api_fluid_layers_conv3d_transpose`


1D序列卷积
==============

Fluid可以表示变长的序列结构，这里的变长是指不同样本的时间步(step)数不一样，通常是一个2D的Tensor和一个能够区分的样本长度的辅助结构来表示。假定，2D的Tensor的形状是shape，shape[0]是所有样本的总时间步数，shape[1]是序列特征的大小。

基于此数据结构的卷积在Fluid里称作序列卷积，也表示一维卷积。同图像卷积，序列卷积的输入参数有卷积核大小、填充大小、滑动步长，但与2D卷积不同的是，这些参数个数都为1。**注意**，目前仅支持stride为1的情况，输出序列的时间步数和输入序列相同。

假如：输入序列形状为(T, N)， T即该序列的时间步数，N是序列特征大小；卷积核的上下文步长为K，输出序列长度为M，则卷积核权重形状为(K * N, M），输出序列形状为(T, M)。

另外，参考DeepSpeech，Fluid实现了行卷积row convolution, 或称
`look ahead convolution <http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf>`_ ，
该卷积相比上述普通序列卷积可以减少参数。


- API汇总:
 - :ref:`cn_api_fluid_layers_sequence_conv`
 - :ref:`cn_api_fluid_layers_row_conv`
