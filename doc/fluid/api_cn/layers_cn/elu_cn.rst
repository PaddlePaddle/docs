.. _cn_api_fluid_layers_elu:

elu
-------------------------------

.. py:function:: paddle.fluid.layers.elu(x, alpha=1.0, name=None)

ELU激活层（ELU Activation Operator）

根据 https://arxiv.org/abs/1511.07289 对输入张量中每个元素应用以下计算。

.. math::
        \\out=max(0,x)+min(0,α∗(e^{x}−1))\\

参数:
 - **x** (Variable) - 该OP的输入为Tensor。数据类型为float32或float64。
 - **alpha** (FLOAT, 可选) - ELU的alpha值，默认值为1.0。
 - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回: ELU操作符的输出

返回类型： Variable - 该OP的输出为Tensor，数据类型为float32，float64。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3,10,32,32], dtype="float32")
    y = fluid.layers.elu(x, alpha=0.2)
