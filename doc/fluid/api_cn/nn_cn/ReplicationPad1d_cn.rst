.. _cn_api_nn_ReplicationPad1d:

ReplicationPad1d
-------------------------------
.. py:class:: paddle.nn.ReplicationPad1d(pad=[0, 0], data_format="NCHW", name=None)

:alias_main: paddle.nn.ReplicationPad1d
:alias: paddle.nn.ReplicationPad1d,paddle.nn.layer.ReplicationPad1d,paddle.nn.common.ReplicationPad1d


**ReplicationPad1d**

按照 pad 对输入 以replicate模式进行 ``pad``，即填充输入的边界值。

参数：
  - **pad** (Variable | List[int32]) - 填充大小。pad的格式为[pad_left, pad_right]。
    默认值为[0, 0]。
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
    data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
    my_pad = nn.ReplicationPad1d(pad=pad)
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        result = my_pad(data)
    print(result.numpy())
    # [[[1. 1. 2. 3. 3. 3.]
    #   [1. 4. 5. 6. 6. 6.]]]
