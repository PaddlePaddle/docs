.. _cn_api_paddle_nn_LSTM:

LSTM
-------------------------------

.. py:class:: paddle.nn.LSTM(input_size, hidden_size, num_layers=1, direction="forward", dropout=0., time_major=False, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, proj_size=0, name=None)



**长短期记忆网络**

长短期记忆网络（LSTM），根据输入序列和给定的初始状态计算返回输出序列和最终状态。在该网络中的每一层对应输入的 step，每个 step 根据当前时刻输入 :math:`x_{t}` 和上一时刻状态 :math:`h_{t-1}, c_{t-1}` 计算当前时刻输出 :math:`y_{t}` 并更新状态 :math:`h_{t}, c_{t}` 。

状态更新公式如下：

..  math::

        i_{t} & = \sigma(W_{ii}x_{t} + b_{ii} + W_{hi}h_{t-1} + b_{hi})

        f_{t} & = \sigma(W_{if}x_{t} + b_{if} + W_{hf}h_{t-1} + b_{hf})

        o_{t} & = \sigma(W_{io}x_{t} + b_{io} + W_{ho}h_{t-1} + b_{ho})

        g_{t} & = \tanh(W_{ig}x_{t} + b_{ig} + W_{hg}h_{t-1} + b_{hg})

        c_{t} & = f_{t} * c_{t-1} + i_{t} * g_{t}

        h_{t} & = o_{t} * \tanh(c_{t})

        y_{t} & = h_{t}

若设置了 `proj_size`，隐状态 :math:`h_{t}` 将会被映射到指定维度：

.. math::
        h_{t} = h_{t} * W_{proj\_size}

其中：
    - :math:`\sigma` ：sigmoid 激活函数。

参数
::::::::::::

    - **input_size** (int) - 输入 :math:`x` 的大小。
    - **hidden_size** (int) - 隐藏状态 :math:`h` 大小。
    - **num_layers** (int，可选) - 循环网络的层数。例如，将层数设为 2，会将两层 GRU 网络堆叠在一起，第二层的输入来自第一层的输出。默认为 1。
    - **direction** (str，可选) - 网络迭代方向，可设置为 forward 或 bidirect（或 bidirectional）。foward 指从序列开始到序列结束的单向 GRU 网络方向，bidirectional 指从序列开始到序列结束，又从序列结束到开始的双向 GRU 网络方向。默认为 forward。
    - **time_major** (bool，可选) - 指定 input 的第一个维度是否是 time steps。如果 time_major 为 True，则 Tensor 的形状为[time_steps, batch_size, input_size]，否则为[batch_size, time_steps, input_size]。`time_steps` 指输入序列的长度。默认为 False。
    - **dropout** (float，可选) - dropout 概率，指的是除第一层外每层输入时的 dropout 概率。范围为[0, 1]。默认为 0。
    - **weight_ih_attr** (ParamAttr，可选) - weight_ih 的参数。默认为 None。
    - **weight_hh_attr** (ParamAttr，可选) - weight_hh 的参数。默认为 None。
    - **bias_ih_attr** (ParamAttr，可选) - bias_ih 的参数。默认为 None。
    - **bias_hh_attr** (ParamAttr，可选) - bias_hh 的参数。默认为 None。
    - **proj_size** (int，可选) - 若大于 0，则会使用投影层将隐状态隐射到指定大小，其值必须小于 `hidden_size` 。默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

输入
::::::::::::

    - **inputs** (Tensor) - 网络输入。如果 time_major 为 True，则 Tensor 的形状为[time_steps,batch_size,input_size]，如果 time_major 为 False，则 Tensor 的形状为[batch_size,time_steps,input_size]。`time_steps` 指输入序列的长度。
    - **initial_states** (tuple，可选) - 网络的初始状态，一个包含 h 和 c 的元组，形状为[num_layers * num_directions, batch_size, hidden_size]。如果没有给出则会以全零初始化。
    - **sequence_length** (Tensor，可选) - 指定输入序列的实际长度，形状为[batch_size]，数据类型为 int64 或 int32。在输入序列中所有 time step 不小于 sequence_length 的元素都会被当作填充元素处理（状态不再更新）。

输出
::::::::::::

    - **outputs** (Tensor) - 输出，由前向和后向 cell 的输出拼接得到。如果 time_major 为 True，则 Tensor 的形状为[time_steps, batch_size, num_directions * hidden_size]，当设置了 `proj_size` 时，输出形状会被映射为[time_steps, batch_size, num_directions * proj_size]，如果 time_major 为 False，则 Tensor 的形状为[batch_size,time_steps,num_directions * hidden_size]，当 direction 设置为 bidirectional 时，num_directions 等于 2，否则等于 1。`time_steps` 指输出序列的长度。
    - **final_states** (tuple) - 最终状态，一个包含 h 和 c 的元组。形状为[num_layers * num_directions, batch_size, hidden_size]，当设置了 `proj_size` 时， h 的形状会被映射为[num_layers * num_directions, batch_size, proj_size]。 当 direction 设置为 bidirectional 时，num_directions 等于 2，返回值的前向和后向的状态的索引是 0，2，4，6..。和 1，3，5，7...，否则等于 1。


代码示例
::::::::::::

COPY-FROM: paddle.nn.LSTM
