.. _cn_api_nn_functional_max_unpool1d:


max_unpool1d
-------------------------------

.. py:function:: paddle.nn.functional.max_unpool1d(x, indices, kernel_size, stride=None, padding=0, data_format="NCL", output_size=None, name=None)

这个API实现了 `1D最大反池化` 操作

.. note::
   更多细节请参考对应的 `Class` 请参考 :ref:`cn_api_nn_MaxUnPool1D` 。


输入：
    X 形状：:math:`(N, C, L_{in})`
输出：
    Output 形状：:math:`(N, C, L_{out})` 具体计算公式为

.. math::
  L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel_size}

或由参数 `output_size` 直接指定


参数
:::::::::
    - **x** (Tensor)：形状为 `[N,C,L]` 的3-D Tensor，N是批尺寸，C是通道数，L是特征长度，数据类型为float32或float64。
    - **indices** (Tensor): 形状为 `[N,C,L]` 的3-D Tensor，N是批尺寸，C是通道数，L是特征长度，数据类型为int32。
    - **kernel_size** (int|list|tuple): 反池化的滑动窗口大小。
    - **stride** (int|list|tuple，可选)：池化层的步长。如果它是一个元组或列表，它必须包含一个整数，(pool_stride_Length)，默认值：None。
    - **padding** (string|int|list|tuple，可选) 池化填充,默认值：0。
    - **output_size** (list|tuple, 可选): 目标输出尺寸。 如果 output_size 没有被设置，则实际输出尺寸会通过(input_shape, kernel_size, stride, padding)自动计算得出，默认值：None。
    - **data_format** (str, 可选)： 输入和输出的数据格式，只能是"NCL"。N是批尺寸，C是通道数，L是特征长度。默认值："NCL"
    - **name** (str，可选)：函数的名字，默认为None.



返回
:::::::::

3-D Tensor，unpooling 的计算结果。


代码示例
:::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F

    data = paddle.rand(shape=[1, 3, 16])
    pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_mask=True)
    # pool_out shape: [1, 3, 8],  indices shape: [1, 3, 8]
    unpool_out = F.max_unpool1d(pool_out, indices, kernel_size=2, padding=0)
    # unpool_out shape: [1, 3, 16]