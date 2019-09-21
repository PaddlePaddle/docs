.. _cn_api_fluid_layers_equal:

equal
-------------------------------

.. py:function:: paddle.fluid.layers.equal(x,y,cond=None)

**equal**
该OP返回 :math:`x==y` 逐元素比较x和y是否相等的结果。

参数：
    - **x** (Variable)- 比较是否相等操作的第一个Tensor，数据类型为 float32, float64，int32, int64
    - **y** (Variable)- 比较是否相等操作的第二个Tensor，数据类型为 float32, float64, int32, int64 
    - **cond** (Variable|None)- 输出Tensor（可选），用来存储比较是否相等操作的结果

返回：返回逐元素比较结果的Tensor， 数据类型bool。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[3,10,32,32], dtype="float32")
    limit = fluid.layers.data(name="limit", shape=[3,10,32,32], dtype="float32")
    less = fluid.layers.equal(x=label,y=limit)




