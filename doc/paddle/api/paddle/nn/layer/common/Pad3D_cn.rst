.. _cn_api_nn_Pad3D:

Pad3D
-------------------------------
.. py:class:: paddle.nn.Pad3D(padding, mode="constant", value=0.0, data_format="NCL", name=None)

**Pad3D**

按照 padding、mode 和 value 属性对输入进行填充。

参数：
  - **padding** (Tensor | List[int32]) - 填充大小。pad的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。
  - **mode** (str) - padding的四种模式，分别为 `'constant'`, `'reflect'`, `'replicate'` 和`'circular'`。
    `'constant'` 表示填充常数 `value`；`'reflect'` 表示填充以input边界值为轴的映射；`'replicate'` 表示
    填充input边界值；`'circular'`为循环填充input。默认值为 `'constant'` 。
  - **value** (float32) - 以 `'constant'` 模式填充区域时填充的值。默认值为0.0。
  - **data_format** (str)  - 指定input的format，可为 'NCDHW' 或者 'NDHWC'，默认值为`'NCDHW'`。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回：无

**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn as nn
    import numpy as np
    input_shape = (1, 1, 1, 2, 3)
    pad = [1, 0, 1, 2, 0, 0]
    mode = "constant"
    data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
    my_pad = nn.Pad3D(padding=pad, mode=mode)
    result = my_pad(data)
    print(result.numpy())
    # [[[[[0. 0. 0. 0.]
    #     [0. 1. 2. 3.]
    #     [0. 4. 5. 6.]
    #     [0. 0. 0. 0.]
    #     [0. 0. 0. 0.]]]]]
