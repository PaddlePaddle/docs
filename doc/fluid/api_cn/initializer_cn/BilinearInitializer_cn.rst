.. _cn_api_fluid_initializer_BilinearInitializer:

BilinearInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.BilinearInitializer

该初始化函数用于转置卷积函数，进行上采样。用户通过任意整型因子放大shape为(B，C，H，W)的特征图。用法如下：

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    factor = 2
    C = 2
    w_attr = fluid.initializer.ParamAttr(
        learning_rate=0.,
        regularizer=fluid.regularizer.L2Decay(0.),
        initializer=fluid.initializer.Bilinear())
    x = fluid.layers.data(name="data", shape=[3, 32, 32],
                          dtype="float32")
    conv_up = fluid.layers.conv2d_transpose(
        input=x,
        num_filters=C,
        output_size=None,
        filter_size=2 * factor - factor % 2,
        padding=int(math.ceil((factor - 1) / 2.)),
        stride=factor,
        groups=C,
        param_attr=w_attr,
        bias_attr=False)

num_filters = C和groups = C 表示这是按通道转置的卷积函数。滤波器shape为(C,1,K,K)，K为filter_size。该初始化函数为滤波器的每个通道设置(K,K)插值核。输出特征图的最终输出shape为(B,C,factor*H,factor*W)。注意学习率和权重衰减设为0，以便在训练过程中双线性插值的系数值保持不变





