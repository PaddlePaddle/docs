.. _cn_api_fluid_layers_stanh:

stanh
-------------------------------

.. py:function:: paddle.fluid.layers.stanh(x, scale_a=0.6666666666666666, scale_b=1.7159, name=None)

STanh 激活算子（STanh Activation Operator.）

.. math::
          \\out=b*\frac{e^{a*x}-e^{-a*x}}{e^{a*x}+e^{-a*x}}\\

参数：
    - **x** (Variable) - STanh operator的输入
    - **scale_a** (FLOAT|2.0 / 3.0) - 输入的a的缩放参数
    - **scale_b** (FLOAT|1.7159) - b的缩放参数
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: STanh操作符的输出

返回类型: 输出(Variable)

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.stanh(x, scale_a=0.67, scale_b=1.72)







