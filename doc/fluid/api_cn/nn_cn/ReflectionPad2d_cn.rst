.. _cn_api_nn_ReflectionPad2d:

ReflectionPad2d
-------------------------------
.. py:class:: paddle.nn.ReflectionPad2d(padding, data_format="NCHW", name=None)

**ReflectionPad2d**

按照 padding 对输入 以reflection模式进行 ``pad``，即填充以输入边界值为轴的映射 。

参数：
  - **padding** (Tensor | List[int32]) - 填充大小。pad的格式为[pad_left, pad_right, pad_top, pad_bottom]。。
  - **data_format** (str)  - 指定input的format，可为 `'NCHW'` 或者 `'NHWC'`，默认值为`'NCHW'`。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回：无

**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn as nn
    import numpy as np
    paddle.disable_static()

    input_shape = (1, 1, 4, 3)
    pad = [1, 0, 1, 2]
    data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
    my_pad = nn.ReflectionPad2d(padding=pad)
    data = paddle.to_tensor(data)
    result = my_pad(data)
    print(result.numpy())
    # [[[[ 5.  4.  5.  6.]
    #    [ 2.  1.  2.  3.]
    #    [ 5.  4.  5.  6.]
    #    [ 8.  7.  8.  9.]
    #    [11. 10. 11. 12.]
    #    [ 8.  7.  8.  9.]
    #    [ 5.  4.  5.  6.]]]]