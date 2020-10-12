.. _cn_api_paddle_nn_layer_rnn_BiRNN:

BiRNN
-------------------------------

.. py:class:: paddle.nn.BiRNN(cell_fw, cell_bw, time_major=False)



**双向循环神经网络**

该OP是双向循环神经网络（BiRNN）的封装，将输入的前向cell和后向cell封装为一个双向循环神经网络。该网络单独执行前向和后向cell的计算并将输出拼接。

参数：
    - **cell_fw** (RNNCellBase) - 前向cell。RNNCellBase类的一个实例。
    - **cell_bw** (RNNCellBase) - 后向cell。RNNCellBase类的一个实例。
    - **time_major** (bool，可选) - 指定input的第一个维度是否是time steps。默认为False。
    
输入:
    - **inputs** (Tensor) - 输入。如果time_major为False，则Tensor的形状为[batch_size,time_steps,input_size]，如果time_major为True，则Tensor的形状为[time_steps,batch_size,input_size]，input_size为cell的input_size。
    - **initial_states** (list|tuple，可选) - 输入前向和后向cell的初始状态，如果没有给出则会调用 :code:`cell.get_initial_states` 生成初始状态。默认为None。
    - **sequence_length** (Tensor，可选) - 指定输入序列的长度，形状为[batch_size]，数据类型为int64或int32。在输入序列中所有time step不小于sequence_length的元素都会被当作填充元素处理（状态不再更新）。

输出:
    - **outputs** (Tensor) - 输出，由前向和后向cell的输出拼接得到。如果time_major为False，则Tensor的形状为[batch_size,time_steps,cell_fw.hidden_size + cell_bw.hidden_size]，如果time_major为True，则Tensor的形状为[time_steps,batch_size,cell_fw.hidden_size + cell_bw.hidden_size]。
    - **final_states** (tuple) - 前向和后向cell的最终状态。
    
.. Note::
    该类是一个封装rnn cell的低级api，用户在使用forward函数时须确保initial_states满足cell的要求。


**代码示例**：

.. code-block:: python

            import paddle

            cell_fw = paddle.nn.LSTMCell(16, 32)
            cell_bw = paddle.nn.LSTMCell(16, 32)

            rnn = paddle.nn.BiRNN(cell_fw, cell_bw)
            inputs = paddle.rand((2, 23, 16))
            outputs, final_states = rnn(inputs)
            
            print(outputs.shape)
            print(final_states[0][0].shape,len(final_states),len(final_states[0]))
            
            #[4,23,64]
            #[2,32] 2 2
