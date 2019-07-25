.. _cn_api_fluid_layers_selu:

selu
-------------------------------

.. py:function:: paddle.fluid.layers.selu(x, scale=None, alpha=None, name=None)

**实现Selu运算**

有如下等式：

.. math::
    selu= \lambda*
    \begin{cases}
         x                      &\quad \text{ if } x>0 \\
         \alpha * e^x - \alpha  &\quad \text{ if } x<=0
    \end{cases}

输入 ``x`` 可以选择性携带LoD信息。输出和它共享此LoD信息(如果有)。

参数:
  - **x** (Variable) – 输入张量
  - **scale** (float, None) – 如果标度没有设置，其默认值为 1.0507009873554804934193349852946。 详情请见： `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515.pdf>`_
  - **alpha** (float, None) – 如果没有设置改参数, 其默认值为 1.6732632423543772848170429916717。 详情请见： `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515.pdf>`_
  - **name** (str|None, default None) – 该层命名，若为None则自动为其命名

返回：一个形和输入张量相同的输出张量

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
     
    input = fluid.layers.data(
         name="input", shape=[3, 9, 5], dtype="float32")

    output = fluid.layers.selu(input)













