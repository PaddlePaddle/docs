.. _cn_api_fluid_layers_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid(x, name=None)




sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数
:::::::::

    - **x** Tensor - 数据类型为float32，float64。激活函数的输入值。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。默认：None

返回
:::::::::
Tensor，激活函数的输出值，数据类型为float32。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.uniform(min=-3., max=3., shape=[3])
    y = paddle.nn.functional.sigmoid(x)
    print(x)
    print(y)









