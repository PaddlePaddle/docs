.. _cn_api_nn_AdaptiveMaxPool1D:


AdaptiveMaxPool1D
-------------------------------

.. py:function:: paddle.nn.AdaptiveMaxPool1D(output_size, return_mask=False, name=None)

该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算1D的自适应最大池化。输入和输出都是3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `L` 是输入特征的长度.

计算公式如下:

..  math::

    lstart &= floor(i * L_{in} / L_{out})

    lend &= ceil((i + 1) * L_{in} / L_{out})

    Output(i) &= max(Input[lstart:lend])


参数
:::::::::
    - **output_size** (int|list|tuple): 算子输出特征图的长度，其数据类型为int,list或tuple。
    - **return_mask** (bool，可选): 如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状
:::::::::
    - **x** (Tensor): 默认形状为（批大小，通道数，输出特征长度），即NCL格式的3-D Tensor。 其数据类型为float32或者float64。
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征长度），即NCL格式的3-D Tensor。 其数据类型与输入x相同。

返回
:::::::::
计算AdaptiveMaxPool1D的可调用对象

抛出异常
:::::::::
    - ``ValueError`` - ``output_size`` 应是一个整数或长度为1的list，tuple

代码示例
:::::::::

.. code-block:: python

        # max adaptive pool1d
        # suppose input data in shape of [N, C, L], `output_size` is m or [m],
        # output shape is [N, C, m], adaptive pool divide L dimension
        # of input data into m grids averagely and performs poolings in each
        # grid to get output.
        # adaptive max pool performs calculations as follow:
        #
        #     for i in range(m):
        #         lstart = floor(i * L / m)
        #         lend = ceil((i + 1) * L / m)
        #         output[:, :, i] = max(input[:, :, lstart: lend])
        #
        import paddle
        import paddle.nn as nn
        import numpy as np
        
        data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
        AdaptiveMaxPool1D = nn.layer.AdaptiveMaxPool1D(output_size=16)
        pool_out = AdaptiveMaxPool1D(data)
        # pool_out shape: [1, 3, 16]

        # for return_mask = true
        AdaptiveMaxPool1D = nn.layer.AdaptiveMaxPool1D(output_size=16, return_mask=True)
        pool_out, indices = AdaptiveMaxPool1D(data)
        # pool_out shape: [1, 3, 16], indices shape: [1, 3, 16]
