.. _cn_api_nn_ConstantPad1d:

ConstantPad1d
-------------------------------
.. py:class:: paddle.nn.ConstantPad1d(padding=[0, 0], value=0.0, data_format="NCL", name=None)

**ConstantPad1d**

按照 padding 对输入 以constant模式进行 ``pad``，即填充固定值。

参数：
  - **padding** (Variable | List[int32]) - 填充大小。pad的格式为[pad_left, pad_right]。
    默认值为[0, 0]。
  - **value** (float32) - 待填充的值，默认值为0.0。
  - **data_format** (str)  - 指定input的format，可为 `'NCL'` 或者 `'NLC'`，默认值为`'NCL'`。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回：无

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.nn as nn
    import numpy as np
    input_shape = (1, 2, 3)
    pad = [1, 2]
    value = 0.0
    data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
    my_pad = nn.ConstantPad1d(padding=pad, value=value)
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        result = my_pad(data)
    print(result.numpy())
    # [[[0. 1. 2. 3. 0. 0.]
    #   [0. 4. 5. 6. 0. 0.]]]
