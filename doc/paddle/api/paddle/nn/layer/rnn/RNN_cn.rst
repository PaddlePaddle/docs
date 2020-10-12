.. _cn_api_paddle_nn_layer_rnn_RNN:

RNN
-------------------------------

.. py:class:: paddle.nn.RNN(cell, is_reverse=False, time_major=False)



**循环神经网络**

该OP是循环神经网络（RNN）的封装，将输入的Cell封装为一个循环神经网络。它能够重复执行 :code:`cell.forward()` 直到遍历完input中的所有Tensor。

参数：
    - **cell** (RNNCellBase) - RNNCellBase类的一个实例。
    - **is_reverse** (bool，可选) - 指定遍历input的方向。默认为False
    - **time_major** (bool，可选) - 指定input的第一个维度是否是time steps。默认为False。
    
输入:
    - **inputs** (Tensor) - 输入（可以是多层嵌套的）。如果time_major为False，则Tensor的形状为[batch_size,time_steps,input_size]，如果time_major为True，则Tensor的形状为[time_steps,batch_size,input_size]，input_size为cell的input_size。
    - **initial_states** (Tensor|list|tuple，可选) - 输入cell的初始状态（可以是多层嵌套的），如果没有给出则会调用 :code:`cell.get_initial_states` 生成初始状态。默认为None。
    - **sequence_length** (Tensor，可选) - 指定输入序列的长度，形状为[batch_size]，数据类型为int64或int32。在输入序列中所有time step不小于sequence_length的元素都会被当作填充元素处理（状态不再更新）。

输出:
    - **outputs** (Tensor|list|tuple) - 输出。如果time_major为False，则Tensor的形状为[batch_size,time_steps,hidden_size]，如果time_major为True，则Tensor的形状为[time_steps,batch_size,hidden_size]。
    - **final_states** (Tensor|list|tuple) - cell的最终状态，嵌套结构，形状和数据类型都与初始状态相同。
    
.. Note::
    该类是一个封装rnn cell的低级api，用户在使用forward函数时须确保initial_states满足cell的要求。


**代码示例**：

.. code-block:: python

            import paddle

            inputs = paddle.rand((4, 23, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.SimpleRNNCell(16, 32)
            rnn = paddle.nn.RNN(cell)
            outputs, final_states = rnn(inputs, prev_h)
            
            print(outputs.shape)
            print(final_states.shape)
            
            #[4,23,32]
            #[4,32]
