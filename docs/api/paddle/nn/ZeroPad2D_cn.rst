.. _cn_api_nn_ZeroPad2D:

ZeroPad2D
-------------------------------
.. py:class:: paddle.nn.ZeroPad2D(self, padding, data_format="NCHW", name=None)

**ZeroPad2D**

按照 padding 属性对输入进行零填充。

参数
:::::::::

  - **padding** (Tensor | List[int] | int]) - 填充大小。如果是int，则在所有待填充边界使用相同的填充，
    否则填充的格式为[pad_left, pad_right, pad_top, pad_bottom]。
  - **data_format** (str)  - 指定输入的format，可为 ``'NCHW'`` 或者 ``'NHWC'``，默认值为 ``'NCHW'``。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回：无

形状
:::::::::

  - x(Tensor): ZeroPadD层的输入，要求形状为4-D，dtype为 ``'float32'`` 或 ``'float64'``
  - output(Tensor): 输出，形状为4-D，dtype与 ``'input'`` 相同

代码示例
:::::::::

.. code-block:: python

    import paddle
    import paddle.nn as nn
    import numpy as np

    input_shape = (1, 1, 2, 3)
    pad = [1, 0, 1, 2]
    data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1

    my_pad = nn.ZeroPad2D(padding=pad)
    result = my_pad(data)

    print(result)
    # [[[[0. 0. 0. 0.]
    #    [0. 1. 2. 3.]
    #    [0. 4. 5. 6.]
    #    [0. 0. 0. 0.]
    #    [0. 0. 0. 0.]]]]


