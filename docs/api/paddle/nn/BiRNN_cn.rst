.. _cn_api_paddle_nn_layer_rnn_BiRNN:

BiRNN
-------------------------------

.. py:class:: paddle.nn.BiRNN(cell_fw, cell_bw, time_major=False)



**双向循环神经网络**

该 OP 是双向循环神经网络（BiRNN）的封装，将输入的前向 cell 和后向 cell 封装为一个双向循环神经网络。该网络单独执行前向和后向 cell 的计算并将输出拼接。

参数
::::::::::::

    - **cell_fw** (RNNCellBase) - 前向 cell。RNNCellBase 类的一个实例。
    - **cell_bw** (RNNCellBase) - 后向 cell。RNNCellBase 类的一个实例。
    - **time_major** (bool，可选) - 指定 input 的第一个维度是否是 time steps。默认为 False。

输入
::::::::::::

    - **inputs** (Tensor) - 输入。如果 time_major 为 False，则 Tensor 的形状为[batch_size,time_steps,input_size]，如果 time_major 为 True，则 Tensor 的形状为[time_steps,batch_size,input_size]，input_size 为 cell 的 input_size。
    - **initial_states** (list|tuple，可选) - 输入前向和后向 cell 的初始状态，如果没有给出则会调用 :code:`cell.get_initial_states` 生成初始状态。默认为 None。
    - **sequence_length** (Tensor，可选) - 指定输入序列的长度，形状为[batch_size]，数据类型为 int64 或 int32。在输入序列中所有 time step 不小于 sequence_length 的元素都会被当作填充元素处理（状态不再更新）。

输出
::::::::::::

    - **outputs** (Tensor) - 输出，由前向和后向 cell 的输出拼接得到。如果 time_major 为 False，则 Tensor 的形状为[batch_size,time_steps,cell_fw.hidden_size + cell_bw.hidden_size]，如果 time_major 为 True，则 Tensor 的形状为[time_steps,batch_size,cell_fw.hidden_size + cell_bw.hidden_size]。
    - **final_states** (tuple) - 前向和后向 cell 的最终状态。

.. note::
    该类是一个封装 rnn cell 的低级 api，用户在使用 forward 函数时须确保 initial_states 满足 cell 的要求。


代码示例
::::::::::::

COPY-FROM: paddle.nn.BiRNN
