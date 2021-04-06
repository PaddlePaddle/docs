.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.cos(x, name=None)




余弦函数。

输入范围是 `(-inf, inf)` ， 输出范围是 `[-1,1]`。

.. math::

    out = cos(x)

参数:
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64 、float16。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        x = paddle.to_tensor([[-1, np.pi], [1, 15.6]], dtype='float32')
        res = paddle.cos(x)
        print(res)
        # [[ 0.54030231 -1.        ]
        # [ 0.54030231 -0.99417763]]
