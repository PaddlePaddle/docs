.. _cn_api_nn_MaxUnPool2D:

MaxPool2D
-------------------------------

.. py:function:: paddle.nn.MaxUnPool2D(kernel_size, stride=None,padding=0,data_format="NCHW",output_size=None,name=None)

该接口用于构建 `MaxUnPool2D` 类的一个可调用对象，根据输入的input和最大值位置计算出池化的逆结果。所有非最大值设置为零。

输入：
    X 形状：:math:`(N, C, H_{in}, W_{in})`
输出：
    Output 形状：:math:`(N, C, H_{out}, W_{out})` 具体计算公式为

.. math::
  H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel_size[0]}

.. math::
  W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel_size[1]}

或由参数 `output_size` 直接指定



参数
:::::::::
    - **kernel_size** (int|list|tuple): 反池化的滑动窗口大小。
    - **stride** (int|list|tuple，可选)：池化层的步长。如果它是一个元组或列表，它必须是两个相等的整数，(pool_stride_Height, pool_stride_Width)，默认值：None。
    - **padding** (string|int|list|tuple，可选) 池化填充,默认值：0。
    - **output_size** (list|tuple, 可选): 目标输出尺寸。 如果 output_size 没有被设置，则实际输出尺寸会通过(input_shape, kernel_size, padding)自动计算得出，默认值：None。
    - **data_format** (str, 可选)： 输入和输出的数据格式，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"
    - **name** (str，可选)：函数的名字，默认为None.



形状
:::::::::
    - **x** (Tensor): 默认形状为（批大小，通道数，高度，宽度），即NCHW格式的4-D Tensor。 其数据类型为float32或float64。
    - **indices** (Tensor): 默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即NCHW格式的4-D Tensor。 其数据类型为float32或float64。
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即NCHW格式的4-D Tensor。其数据类型与输入一致。


返回
:::::::::
计算MaxUnPool2D的可调用对象


代码示例
:::::::::

.. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        data = paddle.rand(shape=[1,1,7,7])
        pool_out, indices = F.max_pool2d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
        # pool_out shape: [1, 1, 3, 3],  indices shape: [1, 1, 3, 3]
        UnPool2D = paddle.nn.MaxUnPool2D(kernel_size=2, padding=0)
        unpool_out = UnPool2D(pool_out, indices)
        # unpool_out shape: [1, 1, 6, 6]
