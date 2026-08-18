.. _cn_api_paddle_nn_utils_rnn_pad_sequence:

pad_sequence
-------------------------------

.. py:function:: paddle.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side='right')

将一组长度不同、其余维度相同的 Tensor 填充到相同长度并堆叠。

参数
:::::::::

    - **sequences** (Iterable[Tensor]) - 形状为 ``[L, *]`` 的 Tensor 序列；各 Tensor 的尾部维度和 dtype 必须相同。
    - **batch_first** (bool，可选) - True 时输出布局为 ``[B, T, *]``，否则为 ``[T, B, *]``。默认值：False。
    - **padding_value** (float，可选) - 填充元素的值。默认值：0.0。
    - **padding_side** (str，可选) - 填充方向，可为 ``'right'`` 或 ``'left'``。默认值：``'right'``。

返回
:::::::::

Tensor，填充并堆叠后的结果。

代码示例
:::::::::

COPY-FROM: paddle.nn.utils.rnn.pad_sequence
