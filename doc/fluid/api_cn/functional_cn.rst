.. _cn_api_fluid_conv2d:

conv2d
-------------------------------

.. py:function:: paddle.fluid.layers.conv2d(input, filter, stride=1, padding=0, dilation=1, groups=None, use_cudnn=True, name=None)

支持直接输入filter的卷积，卷积二维层（convolution2D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、dilations、一组参数计算输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，该值须和输入的通道数保持相同，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : `卷积<http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_.

对每个输入X，有等式：

.. math::

    Out = \sigma \left ( W * X + b \right )

其中：
    - :math:`X` ：输入值，NCHW格式的张量（Tensor）
    - :math:`W` ：滤波器值，MCHW格式的张量（Tensor）
    - :math:`*` ： 卷积操作
    - :math:`\sigma` ：激活函数
    - :math:`Out` ：输出值，``Out`` 和 ``X`` 的shape可能不同

**示例**

- 输入：

  输入shape：:math:`( N,C_{in},H_{in},W_{in} )`

  滤波器shape： :math:`( C_{out},C_{in},H_{f},W_{f} )`

- 输出：

  输出shape： :math:`( N,C_{out},H_{out},W_{out} )`

其中

.. math::

    H_{out} = \frac{\left ( H_{in}+2*paddings[0]-\left ( dilations[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    W_{out} = \frac{\left ( W_{in}+2*paddings[1]-\left ( dilations[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

参数：
    - **input** (Variable) - 格式为[N,C,H,W]格式的输入图像
    - **filter** (int) - 滤波器值。格式为[M, C, H, W]的张量（Tensor）
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常:
  - ``ValueError`` - 如果输入shape和filter_size，stride,padding和group不匹配。

**代码示例**：

.. code-block:: python
    ## paddle 静态图示例
    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[1, 3, 32, 32], dtype='float32', append_batch_size=False)
    filter =  fluid.layers.data(name='filter', shape=[64, 3, 3, 3], dtype='float32', append_batch_size=False)
    out = fluid.conv2d(input=data, filter=filter, groups=1, stride=1, padding=1)
    #out.shape = [1, 64, 32, 32]

    ## paddle 动态图示例
    import paddle.fluid as fluid
    import numpy as np
    data = np.random.random((1, 3, 32, 32)).astype(np.float32)
    filter = np.random.random((64, 3, 3, 3)).astype(np.float32)

    out = fluid.layers.conv2d(input=data, filter=filter, groups=1, stride=1, padding=1)
    #out.shape = [1, 64, 32, 32]
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        filter = fluid.dygraph.to_variable(filter)
        out = fluid.conv2d(input=data, filter=filter, groups=1, stride=1, padding=1)
        # out.shape = [1, 64, 32, 32]

