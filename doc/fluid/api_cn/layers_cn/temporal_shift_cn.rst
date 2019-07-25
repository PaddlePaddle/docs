.. _cn_api_fluid_layers_temporal_shift:

temporal_shift
-------------------------------
.. py:function:: paddle.fluid.layers.temporal_shift(x, seg_num, shift_ratio=0.25, name=None)

**Temporal Shift Operator**

此运算符计算输入（x）的时间移位特征。

输入（x）的形状应为[N*T, C, H, W]，N是批大小，T是 ``seg_num`` 指定的时间段号，C是通道号，H和W是特征的高度和宽度。

时间偏移计算如下：

步骤1：将输入（X）重塑为[N、T、C、H、W]。

步骤2：填充0到第二个(T)尺寸的变形结果，填充宽度每边为1，填充结果的形状为[N，T+2，C，H，W]。

步骤3：假设shift_ratio为1/4，切片填充结果如下：

.. math::

    slice1 &= x[:, :T, :C/4, :, :]

    slice2 &= x[:, 2:T+2, C/4:C/2, :, :]

    slice3 &= x[:, 1:T+1, C/2:, :, :]

步骤4：沿第3(C)维连接三个切片，并将结果重塑为[N*T, C, H, W]。

有关时间移动的详细信息，请参阅文件： `Temporal Shift Module <https://arxiv.org/abs/1811.08383>`_

参数：
  - **x**  (Variable) – 时移算符的输入张量。这是一个4维张量，形状为[N*T，C，H，W]。N为批量大小，T为时间段数，C为信道数，H为特征高度，W为特征宽度
  - **seg_num**  (int) – 时间段编号，这应该是一个正整数。
  - **shift_ratio**  (float) – 通道的移位比、通道的第一个 ``shift_ratio`` 部分沿时间维度移动-1，通道的第二个 ``shift_ratio`` 部分沿时间维度移动1。默认值0.25
  - **name**  (str, default None) – 该层名称

返回：时间移位结果是一个与输入形状和类型相同的张量变量

返回类型：out(Variable)

抛出异常： ``TypeError`` – seg_num 必须是int类型


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name='input', shape=[4,2,2], dtype='float32')
    out = fluid.layers.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)



