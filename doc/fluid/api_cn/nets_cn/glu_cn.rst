.. _cn_api_fluid_nets_glu:

glu
-------------------------------
.. py:function:: paddle.fluid.nets.glu(input, dim=-1)
T
he Gated Linear Units(GLU)由切分（split），sigmoid激活函数和按元素相乘组成。沿着给定维将input拆分成两个大小相同的部分，a和b，计算如下：

.. math::

    GLU(a,b) = a\bigotimes \sigma (b)

参考论文: `Language Modeling with Gated Convolutional Networks <https://arxiv.org/pdf/1612.08083.pdf>`_

参数：
    - **input** (Variable) - 输入变量，张量或者LoDTensor
    - **dim** (int) - 拆分的维度。如果 :math:`dim<0`，拆分的维为 :math:`rank(input)+dim`。默认为-1

返回：变量 —— 变量的大小为输入的一半

返回类型：变量（Variable）

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(
        name="words", shape=[-1, 6, 3, 9], dtype="float32")
    # 输出的形状为[-1, 3, 3, 9]
    output = fluid.nets.glu(input=data, dim=1)  









