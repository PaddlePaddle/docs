.. _cn_api_fluid_layers_increment:

increment
-------------------------------

.. py:function:: paddle.fluid.layers.increment(x, value=1.0, in_place=True)

:alias_main: paddle.increment
:alias: paddle.increment,paddle.tensor.increment,paddle.tensor.math.increment
:old_api: paddle.fluid.layers.increment



使输入Tensor ``x`` 的数据累加 ``value`` , 该OP通常用于循环次数的计数。

参数:
    - **x** (Variable) – 元素个数为1的Tensor，数据类型必须为float32，float64，int32，int64。
    - **value** (float，可选) – 需要增加的值，默认为1.0。
    - **in_place** (bool，可选) – 输出Tensor是否和输入Tensor ``x`` 复用同一块内存，默认为True。

返回：累加计算后的Tensor，形状、数据类型和 ``x`` 一致。

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    counter = fluid.layers.zeros(shape=[1], dtype='float32') # [0.]
    fluid.layers.increment(counter) # [1.]
