.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.fluid.layers.cos(x, name=None)

:alias_main: paddle.cos
:alias: paddle.cos,paddle.tensor.cos,paddle.tensor.math.cos
:old_api: paddle.fluid.layers.cos



余弦函数。

输入范围是 `(-inf, inf)` ， 输出范围是 `[-1,1]`。若输入超出边界则结果为`nan`。

.. math::

    out = cos(x)

参数:
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Tensor

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        x_data = np.array([[-1,np.pi],[1,15.6]]).astype(np.float32)
        x = paddle.to_tensor(x_data)
        res = paddle.cos(x)
        print(res.numpy())
        # [[ 0.54030231 -1.        ]
        # [ 0.54030231 -0.99417763]]
