.. _cn_doc_model:

模型组网
============

完成数据集的构建后，需要构建网络模型。首先介绍飞桨组网相关的API，主要是\ ``paddle.nn``\ 下的API介绍，然后介绍动态图下飞桨框架支持的两种组网方式，分别为 ``Sequential`` 组网与 ``SubClass`` 组网，最后，介绍飞桨框架内置的算法模型。

一、paddle.nn 简介
-------------------------

飞桨框架2.0中，组网相关的API都在\ ``paddle.nn``\ 目录下，你可以通过 ``Sequential`` 或 ``SubClass`` 的方式构建具体的模型。组网相关的API类别与具体的API列表如下表：

+---------------+---------------------------------------------------------------------------+
| 功能          | API名称                                                                   |
+===============+===========================================================================+
| Conv          | Conv1D、Conv2D、Conv3D、Conv1DTranspose、Conv2DTranspose、Conv3DTranspose |
+---------------+---------------------------------------------------------------------------+
| Pool          | AdaptiveAvgPool1D、AdaptiveAvgPool2D、AdaptiveAvgPool3D、                 |
|               | AdaptiveMaxPool1D、AdaptiveMaxPool2D、AdaptiveMaxPool3D、                 |
|               | AvgPool1D、AvgPool2D、AvgPool3D、MaxPool1D、MaxPool2D、MaxPool3D          |
+---------------+---------------------------------------------------------------------------+
| Padding       | Pad1D、Pad2D、Pad3d                                                       |
+---------------+---------------------------------------------------------------------------+
| Activation    | ELU、GELU、Hardshrink、Hardtanh、HSigmoid、LeakyReLU、LogSigmoid、        |
|               | LogSoftmax、PReLU、ReLU、ReLU6、SELU、Sigmoid、Softmax、Softplus、        |
|               | Softshrink、Softsign、Tanh、Tanhshrink                                    |
+---------------+---------------------------------------------------------------------------+
| Normlization  | BatchNorm、BatchNorm1D、BatchNorm2D、BatchNorm3D、GroupNorm、             |
|               | InstanceNorm1D、InstanceNorm2D、InstanceNorm3D、LayerNorm、SpectralNorm、 |
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


二、Sequential 组网
-------------------------

针对顺序的线性网络结构你可以直接使用Sequential来快速完成组网，可以减少类的定义等代码编写。具体代码如下：

.. code:: ipython3

    import paddle
    # Sequential形式组网
    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),
        paddle.nn.Linear(512, 10)
    )

三、SubClass 组网
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

四、飞桨框架内置模型
--------------------------------

你除了可以通过上述方式组建模型外，还可以使用飞桨框架内置的模型，路径为 ``paddle.vision.models`` ，具体列表如下：

.. code:: ipython3

    print('飞桨框架内置模型：', paddle.vision.models.__all__)


.. parsed-literal::

    飞桨框架内置模型： ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV1', 'mobilenet_v1', 'MobileNetV2', 'mobilenet_v2', 'LeNet']

使用方式如下：

.. code:: ipython3

    lenet = paddle.vision.models.LeNet()


你可以通过\ ``paddle.summary()``\ 方法查看模型的结构与每一层输入输出形状，具体如下：

.. code:: ipython3

    paddle.summary(lenet, (64, 1, 28, 28))


.. parsed-literal::

    ---------------------------------------------------------------------------
     Layer (type)       Input Shape          Output Shape         Param #
    ===========================================================================
       Conv2D-1      [[64, 1, 28, 28]]     [64, 6, 28, 28]          60
        ReLU-1       [[64, 6, 28, 28]]     [64, 6, 28, 28]           0
      MaxPool2D-1    [[64, 6, 28, 28]]     [64, 6, 14, 14]           0
       Conv2D-2      [[64, 6, 14, 14]]     [64, 16, 10, 10]        2,416
        ReLU-2       [[64, 16, 10, 10]]    [64, 16, 10, 10]          0
      MaxPool2D-2    [[64, 16, 10, 10]]     [64, 16, 5, 5]           0
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
