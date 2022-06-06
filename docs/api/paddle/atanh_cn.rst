.. _cn_api_fluid_layers_atanh:

atanh
-------------------------------

.. py:function:: paddle.atanh(x, name=None)

Arctanh函数。

.. math::
    out = atanh(x)

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
:::::::::
输出Tensor，与 ``x`` 维度相同、数据类型相同。



代码示例
:::::::::

.. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.atanh(x)
        print(out)
        # [-0.42364895, -0.20273256,  0.10033535,  0.30951962]
