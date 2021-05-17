.. _cn_api_fluid_layers_increment:

increment
-------------------------------

.. py:function:: paddle.increment(x, value=1.0, name=None)


使输入Tensor ``x`` 的数据累加 ``value`` , 该OP通常用于循环次数的计数。

参数:
    - **x** (Tensor) – 元素个数为1的Tensor，数据类型必须为float32，float64，int32，int64。
    - **value** (float，可选) – 需要增加的值，默认为1.0。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor，累加结果，形状、数据类型和 ``x`` 一致。

**代码示例**

..  code-block:: python

    import paddle

    data = paddle.zeros(shape=[1], dtype='float32')
    counter = paddle.increment(data)
    # [1.]
