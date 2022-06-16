.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.abs(x, name=None)




绝对值函数。

.. math::
    out = |x|

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

.. code-block:: python

        import paddle
        
        x = paddle.to_tensor([-1, -2, -3, -4], dtype='float32')
        res = paddle.abs(x)
        print(res)
        # [1, 2, 3, 4]
