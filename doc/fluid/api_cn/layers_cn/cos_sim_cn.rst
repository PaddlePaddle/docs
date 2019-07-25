.. _cn_api_fluid_layers_cos_sim:

cos_sim
-------------------------------

.. py:function:: paddle.fluid.layers.cos_sim(X, Y)

余弦相似度算子（Cosine Similarity Operator）

.. math::

        Out = \frac{X^{T}*Y}{\sqrt{X^{T}*X}*\sqrt{Y^{T}*Y}}

输入X和Y必须具有相同的shape，除非输入Y的第一维为1(不同于输入X)，在计算它们的余弦相似度之前，Y的第一维会被broadcasted，以匹配输入X的shape。

输入X和Y都携带或者都不携带LoD(Level of Detail)信息。但输出仅采用输入X的LoD信息。

参数：
    - **X** (Variable) - cos_sim操作函数的一个输入
    - **Y** (Variable) - cos_sim操作函数的第二个输入

返回：cosine(X,Y)的输出

返回类型：变量（Variable)

**代码示例**

..  code-block:: python

     import paddle.fluid as fluid
     x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
     y = fluid.layers.data(name='y', shape=[1, 7], dtype='float32', append_batch_size=False)
     out = fluid.layers.cos_sim(x, y)





