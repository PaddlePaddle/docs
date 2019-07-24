.. _cn_api_fluid_layers_add_position_encoding:

add_position_encoding
-------------------------------

.. py:function:: paddle.fluid.layers.add_position_encoding(input, alpha, beta, name=None)

**添加位置编码层**

接受形状为[N×M×P]的三维输入张量，并返回一个形为[N×M×P]的输出张量，且输出张量具有位置编码值。

可参考论文: `Attention Is All You Need <http://arxiv.org/pdf/1706.03762.pdf>`_

.. math::

  PE(pos, 2i) &= \sin{(pos / 10000^{2i / P})}\\
  PE(pos, 2i + 1) &= \cos{(pos / 10000^{2i / P})}\\
  Out(:, pos, i) &= \alpha * input(:, pos, i) + \beta * PE(pos, i)

其中:
    - PE(pos, 2i): 偶数位置上数字的增量
    - PE(pos, 2i + 1): 奇数位置上数字的增量

参数:
    - **input**  (Variable) – 形状为[N x M x P]的三维输入张量
    - **alpha**  (float) – 输入张量的倍数
    - **beta**  (float) – 位置编码张量Positional Encoding Tensor的倍数
    - **name**  (string) – 位置编码层的名称


返回:  具有位置编码的三维形状张量[N×M×P]

返回类型: Variable

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
     
  tensor = fluid.layers.data(
        name='tensor',
        shape=[32, 64, 512],
        dtype='float32',
        append_batch_size=False)
  position_tensor = fluid.layers.add_position_encoding(
        input=tensor, alpha=1.0, beta=1.0)











