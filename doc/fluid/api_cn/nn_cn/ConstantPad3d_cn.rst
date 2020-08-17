.. _cn_api_nn_ConstantPad3d:

ConstantPad3d
-------------------------------
.. py:class:: paddle.nn.ConstantPad3d(padding, value=0.0, data_format="NCDHW", name=None)

**ConstantPad3d**

按照 padding 对输入 以constant模式进行 ``pad``，即填充固定值。

参数：
  - **padding** (Tensor | List[int32]) - 填充大小。pad的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。
  - **value** (float32) - 待填充的值，默认值为0.0。
  - **data_format** (str)  - 指定input的format，可为 `'NCDHW'` 或者 `'NDHWC'`，默认值为`'NCDHW'`。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回：无

**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn as nn
    import numpy as np
    paddle.disable_static()

    input_shape = (1, 1, 1, 2, 3)
    pad = [1, 0, 1, 2, 0, 0]
    data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
    my_pad = nn.ConstantPad3d(padding=pad)
    data = paddle.to_tensor(data)
    result = my_pad(data)
    print(result.numpy())
    # [[[[[0. 0. 0. 0.]
    #     [0. 1. 2. 3.]
    #     [0. 4. 5. 6.]
    #     [0. 0. 0. 0.]
    #     [0. 0. 0. 0.]]]]]
