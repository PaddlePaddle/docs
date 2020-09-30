.. _cn_api_fluid_layers_reciprocal:

reciprocal
-------------------------------

.. py:function:: paddle.fluid.layers.reciprocal(x, name=None)




reciprocal 对输入Tensor取倒数


.. math::
    out = \frac{1}{x}

参数:

    - **x** - 输入的多维Tensor,支持的数据类型为float32，float64。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回： 对输入取倒数得到的Tensor，输出Tensor数据类型和维度与输入相同。

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        x_data = np.array([1, 2, 3, 4]).astype(np.float32)
        x = paddle.imperative.to_variable(x_data)
        res = paddle.%s(x)
        print(res.numpy())




