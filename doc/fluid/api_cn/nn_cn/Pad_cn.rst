.. _cn_api_nn_Pad:

Pad
-------------------------------
.. py:class:: paddle.nn.Pad(pad=[0, 0, 0, 0], mode='constant', value=0.0, data_format="NCHW", name=None)

:alias_main: paddle.nn.Pad
:alias: paddle.nn.Pad,paddle.nn.layer.Pad,paddle.nn.common.Pad
:update_api: paddle.fluid.layers.pad



**Pad**

按照 pad 和 mode 属性对input进行 ``pad`` 。

参数：
  - **pad** (Variable | List[int32]) - 填充大小。为List时
    1. 当输入维度为3时，pad的格式为[pad_left, pad_right]。
    2. 当输入维度为4时，pad的格式为[pad_left, pad_right, pad_top, pad_bottom]。
    3. 当输入维度为5时，pad的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。
    默认值为[0, 0, 0, 0]。
  - **mode** (str) - padding的四种模式，分别为 `'constant'`, `'reflect'`, `'replicate'` 和`'circular'`。
    1. `'constant'` 为填充常数 `pad_value`
    2. `'reflect'` 为填充以input边界值为轴的映射
    3. `'replicate'` 为填充input边界值
    4. `'circular' `为循环填充input
  - **value** (float32) - 以 `'constant'` 模式填充区域时填充的值。默认值为0.0。
  - **data_format** (str)  - 指定input的format，可为 `'NCL'`, `'NLC'`, `'NCHW'`, `'NHWC'`, `'NCDHW'`
    或 `'NDHWC'`，默认值为`'NCHW'`。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回：无

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.nn as nn
    import numpy as np
    data = np.ones((1, 1, 2, 2)).astype('float32')
    my_pad = nn.Pad(pad=[1, 1, 1, 1])
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        result = my_pad(data)
    print(result.numpy())
    # [[[[0. 0. 0. 0.]
    #    [0. 1. 1. 0.]
    #    [0. 1. 1. 0.]
    #    [0. 0. 0. 0.]]]]
