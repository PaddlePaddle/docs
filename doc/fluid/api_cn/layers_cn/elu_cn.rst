.. _cn_api_fluid_layers_elu:

elu
-------------------------------

.. py:function:: paddle.fluid.layers.elu(x, alpha=1.0, name=None)

ELU激活层（ELU Activation Operator）

根据 https://arxiv.org/abs/1511.07289 对输入张量中每个元素应用以下计算。

.. math::
        \\out=max(0,x)+min(0,α∗(ex−1))\\

参数:
    - x(Variable)- ELU operator的输入
    - alpha(FAOAT|1.0)- ELU的alpha值
    - name (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: ELU操作符的输出

返回类型: 输出(Variable)

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.elu(x, alpha=0.2)







