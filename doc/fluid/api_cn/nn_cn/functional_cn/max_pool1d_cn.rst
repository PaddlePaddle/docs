.. _cn_api_nn_functional_max_pool1d:


max_pool1d
-------------------------------

.. py:function:: paddle.nn.functional.max_pool1d(x, kernel_size, stride=None, padding=0, return_indices=False, ceil_mode=False, name=None)

该算子根据输入 `x` , `kernel_size` 等参数对一个输入Tensor计算1D的最大值池化。输入和输出都是3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `L` 是输入特征的长度。

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_MaxPool1d` 。


参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, L]` 的3-D Tensor。其中 `N` 是batch size, `C` 是通道数, `L` 是输入特征的长度。 其数据类型为float32或者float64。
    - **kernel_size** (int|list|tuple): 池化核的尺寸大小. 如果kernel_size为list或tuple类型, 其必须包含一个整数.
    - **stride** (int|list|tuple): 池化操作步长. 如果stride为list或tuple类型, 其必须包含一个整数.
    - **padding** (string|int|list|tuple): 池化补零的方式. 如果padding是一个字符串，则必须为 `SAME` 或者 `VALID` 。如果是turple或者list类型， 则应是 `[pad_left, pad_right]` 形式。如果padding是一个非0值，那么表示会在输入的两端都padding上同样长度的0。
    - **return_indices** (bool): 是否返回最大值的索引，默认为False。
    - **ceil_mode** (bool): 是否用ceil函数计算输出的height和width，如果设置为False, 则使用floor函数来计算，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。


返回
:::::::::
``Tensor``, 输入 `x` 经过最大值池化计算得到的目标3-D Tensor，其数据类型与输入相同。


抛出异常
:::::::::
    - ``ValueError`` - 如果 ``padding`` 是字符串但不是 "SAME" 和 "VALID" 。
    - ``ValueError`` - 如果 ``padding`` 是 "VALID" 但 `ceil_mode` 被设置为True。
    - ``ValueError`` - 如果 ``padding`` 是一个长度大于1的list或turple。
    - ``ShapeError`` - 如果输入x不是一个3-D Tensor。
    - ``ShapeError`` - 如果计算得到的输出形状小于等于0。


代码示例
:::::::::

.. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np
        paddle.disable_static()

        data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
        pool_out = F.max_pool1d(data, kernel_size=2, stride=2, padding=0)
        # pool_out shape: [1, 3, 16]

        pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_indices=True)
        # pool_out shape: [1, 3, 16],  indices shape: [1, 3, 16]
