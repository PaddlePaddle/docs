.. _cn_api_fluid_layers_increment:

increment
-------------------------------

.. py:function:: paddle.fluid.layers.increment(x, value=1.0, in_place=True)


该函数为输入 ``x`` 增加 ``value`` 大小, ``value`` 即函数中待传入的参数。该函数默认直接在原变量 ``x`` 上进行运算。

.. note::
    ``x`` 中元素个数必须为1

参数:
    - **x** (Variable|list) – 含有输入值的张量(tensor)
    - **value** (float) – 需要增加在 ``x`` 变量上的值
    - **in_place** (bool) – 判断是否在x变量本身执行操作，True原地执行，False时，返回增加后的副本

返回： 每个元素增加后的对象

返回类型：变量(variable)

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[1], dtype='float32',
                         append_batch_size=False)
    data = fluid.layers.increment(x=data, value=3.0, in_place=True)











