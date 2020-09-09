.. _cn_api_nn_functional_adaptive_max_pool1d:


adaptive_max_pool1d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_max_pool1d(x, output_size, return_indices=False, name=None)

该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算1D的自适应最大值池化。输入和输出都是3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `L` 是输入特征的长度.

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_AdaptiveMaxPool1d` 。


参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, L]` 的3-D Tensor。其中 `N` 是batch size, `C` 是通道数, `L` 是输入特征的长度。 其数据类型为float32或者float64。
    - **output_size** (int|list|tuple): 算子输出特征图的长度，其数据类型为int或list，tuple。
    - **return_indices** (bool): 如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输入 `x` 经过自适应池化计算得到的目标3-D Tensor，其数据类型与输入相同。

抛出异常
:::::::::
    - ``ValueError`` - 如果 ``output_size`` 不是int类型值。

代码示例
:::::::::

.. code-block:: python

        # max adaptive pool1d
        # suppose input data in shape of [N, C, L], `output_size` is m,
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
        import paddle.nn.functional as F
        import numpy as np
        paddle.disable_static()

        data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
        pool_out = F.adaptive_max_pool1d(data, output_size=16)
        # pool_out shape: [1, 3, 16])
