.. _cn_api_paddle_nn_RNN:

RNN
-------------------------------

.. py:class:: paddle.nn.RNN(cell, is_reverse=False, time_major=False)



**循环神经网络**

循环神经网络（RNN）的封装，将输入的 Cell 封装为一个循环神经网络。它能够重复执行 :code:`cell.forward()` 直到达到 inputs 的最大长度。

参数
::::::::::::

    - **cell** (RNNCellBase) - RNNCellBase 类的一个实例。
    - **is_reverse** (bool，可选) - 指定遍历 input 的方向。默认为 False。
    - **time_major** (bool，可选) - 指定 input 的第一个维度是否是 time steps。默认为 False。

输入
::::::::::::

    - **inputs** (Tensor) - 输入（可以是多层嵌套的）。如果 time_major 为 False，则 Tensor 的形状为[batch_size,time_steps,input_size]，如果 time_major 为 True，则 Tensor 的形状为[time_steps,batch_size,input_size]，input_size 为 cell 的 input_size。
    - **initial_states** (Tensor|list|tuple，可选) - 输入 cell 的初始状态（可以是多层嵌套的），如果没有给出则会调用 :code:`cell.get_initial_states` 生成初始状态。默认为 None。
    - **sequence_length** (Tensor，可选) - 指定输入序列的长度，形状为[batch_size]，数据类型为 int64 或 int32。在输入序列中所有 time step 不小于 sequence_length 的元素都会被当作填充元素处理（状态不再更新）。

输出
::::::::::::

    - **outputs** (Tensor|list|tuple) - 输出。如果 time_major 为 False，则 Tensor 的形状为[batch_size,time_steps,hidden_size]，如果 time_major 为 True，则 Tensor 的形状为[time_steps,batch_size,hidden_size]。
    - **final_states** (Tensor|list|tuple) - cell 的最终状态，嵌套结构，形状和数据类型都与初始状态相同。

.. note::
    该类是一个封装 rnn cell 的底层 api，用户在使用 forward 函数时须确保 initial_states 满足 cell 的要求。


代码示例
::::::::::::

COPY-FROM: paddle.nn.RNN
