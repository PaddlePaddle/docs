.. _cn_api_fluid_layers_reverse:

reverse
-------------------------------

.. py:function:: paddle.fluid.layers.reverse(x,axis)

**reverse**

该功能将给定轴上的输入‘x’逆序

参数：
  - **x** (Variable)-预逆序的输入
  - **axis** (int|tuple|list) - 元素逆序排列的轴。如果该参数是一个元组或列表，则对该参数中每个元素值所指定的轴上进行逆序运算。

返回：逆序的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="data", shape=[4, 8], dtype="float32")
        out = fluid.layers.reverse(x=data, axis=0)
        # or:
        out = fluid.layers.reverse(x=data, axis=[0,1])









