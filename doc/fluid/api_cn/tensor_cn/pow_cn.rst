.. _cn_api_tensor_argmax:

pow
-------------------------------

.. py:function:: paddle.pow(x, y, name=None):


该OP是指数激活算子：

.. math::
        out = x^{y}

参数：
    - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``float32`` 或 ``float64`` 或 ``int32`` 或 ``int64``。
    - **y** （Tensor）- 多维 ``Tensor``，数据类型为 ``float32`` 或 ``float64`` 或 ``int32`` 或 ``int64``。Pow OP的指数因子。默认值：1.0。
    - **name** (str) - 默认值None，输出的名称。该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_name`。

返回： Tensor，数据类型和input ``x`` 一致。

**代码示例：**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    
    # 示例1: 参数y是个浮点数
    x_data = np.array([1, 2, 3])
    y = 2
    x = paddle.to_tensor(x_data)
    res = paddle.pow(x, y)
    # print(res.numpy()) # [1 4 9]

    # 示例2: 参数y是个Tensor
    y = paddle.fill_constant(shape=[1], value=2, dtype='float32')
    res = paddle.pow(x, y)
    print(res.numpy()) # [1 4 9]


