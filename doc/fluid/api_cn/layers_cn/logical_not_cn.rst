.. _cn_api_fluid_layers_logical_not:

logical_not
-------------------------------

.. py:function:: paddle.fluid.layers.logical_not(x, out=None, name=None)

logical_not算子

它在X上以元素方式操作，并返回Out。X和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math::
        Out = !X

参数：
        - **x** （Variable）- （LoDTensor）logical_not算子的操作数
        - **out** （Tensor）- 输出逻辑运算的张量。
        - **name** （basestring | None）- 输出的名称。

返回：        (LoDTensor)n维布尔张量。

返回类型：        输出（Variable）。


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    left = fluid.layers.data(
        name='left', shape=[1], dtype='int32')
    result = fluid.layers.logical_not(x=left)







