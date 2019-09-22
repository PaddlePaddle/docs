.. _cn_api_fluid_layers_pow:

pow
-------------------------------

.. py:function:: paddle.fluid.layers.pow(x, factor=1.0, name=None)

该OP是指数激活算子：

.. math::

    out = x^{factor}

**注意：如果使用** ``elementwise_pow``，**请查看相关文档** :ref:`cn_api_fluid_layers_elementwise_pow` 。

参数：
    - **x** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor`` ，数据类型为 ``float32`` 或 ``float64`` 。
    - **factor** （float32|Variable，可选）- ``float32`` 或形状为[1]的 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``。Pow OP的指数因子。默认值：1.0。
    - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`。默认值： ``None``。

返回：维度与输入 `x` 相同的 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``x`` 相同。

返回类型：Variable。


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")

    # example 1: argument factor is float
    y_1 = fluid.layers.pow(x, factor=2.0)
    # y_1 is x^{2.0}

    # example 2: argument factor is Variable
    factor_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
    y_2 = fluid.layers.pow(x, factor=factor_tensor)
    # y_2 is x^{2.0}






