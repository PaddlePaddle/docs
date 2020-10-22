.. _cn_doc_model:

模型组网
============

完成数据集的构建后，需要构建网络模型。本文首先介绍飞桨组网相关的API，主要是\ ``paddle.nn``\ 下的API介绍，然后介绍动态图下飞桨框架支持的两种组网方式，分别为 Sequential组网与 SubClass组网，最后，介绍飞桨框架内置的算法模型。

1. paddle.nn 简介
-------------------------

飞桨框架2.0中，组网相关的API都在\ ``paddle.nn``\ 目录下，您可以通过 Sequential 或 SubClass的方式构建具体的模型。组网相关的API类别与具体的API列表如下表：

+---------------+---------------------------------------------------------------------------+
| 功能          | API名称                                                                   |
+===============+===========================================================================+
| Conv          | Conv1d、Conv2d、Conv3d、ConvTranspose1d、ConvTranspose2d、ConvTranspose3d |
+---------------+---------------------------------------------------------------------------+
| Pool          | AdaptiveAvgPool1d、AdaptiveAvgPool2d、AdaptiveAvgPool3d、                 |
|               | AdaptiveMaxPool1d、AdaptiveMaxPool2d、AdaptiveMaxPool3d、                 |
|               | AvgPool1d、AvgPool2d、AvgPool3d、MaxPool1d、MaxPool2d、MaxPool3d          |
+---------------+---------------------------------------------------------------------------+
| Padding       | ConstantPad1d、ConstantPad2d、ConstantPad3d、Pad2D、ReflectionPad1d、     |
|               | ReflectionPad2d、ReplicationPad1d、ReplicationPad2d、ReplicationPad3d、   |
|               | ZeroPad2d                                                                 |
+---------------+---------------------------------------------------------------------------+
| Activation    | ELU、GELU、Hardshrink、Hardtanh、HSigmoid、LeakyReLU、LogSigmoid、        |
|               | LogSoftmax、PReLU、ReLU、ReLU6、SELU、Sigmoid、Softmax、Softplus、        |
|               | Softshrink、Softsign、Tanh、Tanhshrink                                    |
+---------------+---------------------------------------------------------------------------+
| Normlization  | BatchNorm、BatchNorm1d、BatchNorm2d、BatchNorm3d、GroupNorm、InstanceNorm |
|               | InstanceNorm1d、InstanceNorm2d、InstanceNorm3d、LayerNorm、SpectralNorm、 |
|               | SyncBatchNorm                                                             |
+---------------+---------------------------------------------------------------------------+
| Recurrent NN  | BiRNN、GRU、GRUCell、LSTM、LSTMCell、RNN、RNNCellBase、SimpleRNN、        |
|               | SimpleRNNCell                                                             | 
+---------------+---------------------------------------------------------------------------+
| Transformer   | Transformer、TransformerDecoder、TransformerDecoderLayer、                |
|               | TransformerEncoder、TransformerEncoderLayer                               |
+---------------+---------------------------------------------------------------------------+
| Dropout       | AlphaDropout、Dropout、Dropout2d、Dropout3d                               |
+---------------+---------------------------------------------------------------------------+
| Loss          | BCELoss、BCEWithLogitsLoss、CrossEntropyLoss、CTCLoss、KLDivLoss、L1Loss  |
|               | MarginRankingLoss、MSELoss、NLLLoss、SmoothL1Loss                         |
+---------------+---------------------------------------------------------------------------+


2. Sequential 组网
-------------------------

针对顺序的线性网络结构我们可以直接使用Sequential来快速完成组网，可以减少类的定义等代码编写。具体代码如下：

.. code:: ipython3

    # Sequential形式组网
    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),
        paddle.nn.Linear(512, 10)
    )

3. SubClass 组网
-------------------------

针对一些比较复杂的网络结构，就可以使用Layer子类定义的方式来进行模型代码编写，在\ ``__init__``\ 构造函数中进行组网Layer的声明，在\ ``forward``\ 中使用声明的Layer变量进行前向计算。子类组网方式也可以实现sublayer的复用，针对相同的layer可以在构造函数中一次性定义，在forward中多次调用。

.. code:: ipython3

    # Layer类继承方式组网
    class Mnist(paddle.nn.Layer):
        def __init__(self):
            super(Mnist, self).__init__()

            self.flatten = paddle.nn.Flatten()
            self.linear_1 = paddle.nn.Linear(784, 512)
            self.linear_2 = paddle.nn.Linear(512, 10)
            self.relu = paddle.nn.ReLU()
            self.dropout = paddle.nn.Dropout(0.2)

        def forward(self, inputs):
            y = self.flatten(inputs)
            y = self.linear_1(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.linear_2(y)

            return y

    mnist_2 = Mnist()

4. 飞桨框架内置模型
--------------------------------

在飞桨框架中，除了自己通过上述方式组建模型外，飞桨框架还内置了部分模型，路径为 ``paddle.vision.models`` ，具体列表如下：

.. code:: ipython3

    print('飞桨框架内置模型：', paddle.vision.models.__all__)


.. parsed-literal::

    飞桨框架内置模型：['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'LeNet']

使用方式如下：

.. code:: ipython3

    import paddle
    lenet = paddle.vision.models.Lenet()


我们可以通过\ ``paddle.summary()``\ 方法查看模型的结构与每一层输入输出形状，具体如下：

.. code:: ipython3

    paddle.summary(lenet, (64, 1, 28, 28))


.. parsed-literal::

    ---------------------------------------------------------------------------
     Layer (type)       Input Shape          Output Shape         Param #
    ===========================================================================
       Conv2d-1      [[64, 1, 28, 28]]     [64, 6, 28, 28]          60
        ReLU-1       [[64, 6, 28, 28]]     [64, 6, 28, 28]           0
      MaxPool2d-1    [[64, 6, 28, 28]]     [64, 6, 14, 14]           0
       Conv2d-2      [[64, 6, 14, 14]]     [64, 16, 10, 10]        2,416
        ReLU-2       [[64, 16, 10, 10]]    [64, 16, 10, 10]          0
      MaxPool2d-2    [[64, 16, 10, 10]]     [64, 16, 5, 5]           0
       Linear-1         [[64, 400]]           [64, 120]           48,120
       Linear-2         [[64, 120]]            [64, 84]           10,164
       Linear-3          [[64, 84]]            [64, 10]             850
    ===========================================================================
    Total params: 61,610
    Trainable params: 61,610
    Non-trainable params: 0
    ---------------------------------------------------------------------------
    Input size (MB): 0.19
    Forward/backward pass size (MB): 7.03
    Params size (MB): 0.24
    Estimated Total Size (MB): 7.46
    ---------------------------------------------------------------------------
