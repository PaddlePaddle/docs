.. _cn_api_nn_ReLU:

ReLU
-------------------------------
.. py:class:: paddle.nn.ReLU(inplace=False)

:alias_main: paddle.nn.ReLU
:alias: paddle.nn.ReLU,paddle.nn.layer.ReLU,paddle.nn.layer.activation.ReLU




**ReLU（Rectified Linear Unit）激活层：**

.. math::

        \\Out = max(X, 0)\\

其中，:math:`X` 为输入的 Tensor

参数:
    - **inplace** （bool，可选）- 如果 ``inplace`` 为 ``True``，则 ``ReLU`` 的输入和输出是同一个变量，否则 ``ReLU`` 的输入和输出是不同的变量。默认值：``False``。请注意，如果 ``ReLU`` 的输入同时是其它OP的输入，则 ``inplace`` 必须为False。

返回：无

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.nn as nn
    import numpy as np

    data = np.array([-2, 0, 1]).astype('float32')
    my_relu = nn.ReLU()
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        res = my_relu(data)  # [0, 0, 1]
