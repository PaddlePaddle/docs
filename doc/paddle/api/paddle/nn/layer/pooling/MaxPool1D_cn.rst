.. _cn_api_nn_MaxPool1D:


MaxPool1D
-------------------------------

.. py:function:: paddle.nn.MaxPool1D(kernel_size, stride=None, padding=0, return_mask=False, ceil_mode=False, name=None)

该算子根据输入 `x` , `kernel_size` 等参数对一个输入Tensor计算1D的最大值池化。输入和输出都是3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `L` 是输入特征的长度。

假设输入形状是(N, C, L)，输出形状是 (N, C, L_{out})，卷积核尺寸是k, 1d最大值池化计算公式如下:

..  math::

    Output(N_i, C_i, l) =  max(Input[N_i, C_i, stride \times l:stride \times l+k])

参数
:::::::::
    - **kernel_size** (int|list|tuple): 池化核的尺寸大小. 如果kernel_size为list或tuple类型, 其必须包含一个整数, 最终池化核的大小为该数值。
    - **stride** (int|list|tuple): 池化操作步长. 如果stride为list或tuple类型, 其必须包含一个整数，最终池化操作的步长为该数值。
    - **padding** (string|int|list|tuple): 池化补零的方式. 如果padding是一个字符串，则必须为 `SAME` 或者 `VALID` 。 如果是turple或者list类型， 则应是 `[pad_left, pad_right]` 形式。如果padding是一个非0值，那么表示会在输入的两端都padding上同样长度的0。
    - **return_mask** (bool): 是否返回最大值的索引，默认为False。
    - **ceil_mode** (bool): 是否用ceil函数计算输出的height和width，如果设置为False, 则使用floor函数来计算，默认为False。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。


形状
:::::::::
    - **x** (Tensor): 默认形状为（批大小，通道数，长度），即NCL格式的3-D Tensor。 其数据类型为float32或float64.
    - **output** (Tensor): 默认形状为（批大小，通道数，输出特征长度），即NCL格式的3-D Tensor。 其数据类型与输入x相同。

返回
:::::::::
计算MaxPool1D的可调用对象



代码示例
:::::::::

.. code-block:: python

        import paddle
        import paddle.nn as nn

        data = paddle.uniform(shape=[1, 2, 32], dtype='float32', min=-1, max=1)
        MaxPool1D = nn.layer.MaxPool1D(kernel_size=2, stride=2, padding=0)
        pool_out = MaxPool1D(data)
        # pool_out shape: [1, 3, 16]
        
        MaxPool1D = nn.layer.MaxPool1D(kernel_size=2, stride=2, padding=0, return_mask=True)
        pool_out, indices = MaxPool1D(data)
        # pool_out shape: [1, 3, 16], indices shape: [1, 3, 16]
