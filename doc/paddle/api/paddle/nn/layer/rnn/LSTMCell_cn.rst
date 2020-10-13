.. _cn_api_paddle_nn_layer_rnn_LSTMCell:

LSTMCell
-------------------------------

.. py:class:: paddle.nn.LSTMCell(input_size, hidden_size, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)



**长短期记忆网络单元**

该OP是长短期记忆网络单元（LSTMCell），根据当前时刻输入x（t）和上一时刻状态h（t-1）计算当前时刻输出y（t）并更新状态h（t）。

状态更新公式如下：

..  math::

        i_{t} &= \sigma (W_{ii}x_{t} + b_{ii} + W_{hi}h_{t-1} + b_{hi})\\ 
        f_{t} &= \sigma (W_{if}x_{t} + b_{if} + W_{hf}h_{t-1} + b_{hf})\\ 
        o_{t} &= \sigma (W_{io}x_{t} + b_{io} + W_{ho}h_{t-1} + b_{ho})\\ 
        g_{t} &= \tanh (W_{ig}x_{t} + b_{ig} + W_{hg}h_{t-1} + b_{hg})\\ 
        c_{t} &= f_{t} * c_{t-1} + i_{t} * g_{t}\\ 
        h_{t} &= o_{t} * \tanh (c_{t})\\ 
        y_{t} &= h_{t} 



其中：
    - :math:`\sigma` ：sigmoid激活函数。
   
详情请参考论文 :`An Empirical Exploration of Recurrent Network Architectures <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_。


参数：
    - **input_size** (int) - 输入的大小。
    - **hidden_size** (int) - 隐藏状态大小。
    - **weight_ih_attr** (ParamAttr，可选) - weight_ih的参数。默认为None。
    - **weight_hh_attr** (ParamAttr，可选) - weight_hh的参数。默认为None。
    - **bias_ih_attr** (ParamAttr，可选) - bias_ih的参数。默认为None。
    - **bias_hh_attr** (ParamAttr，可选) - bias_hh的参数。默认为None。
    - **name** (str, 可选): OP的名字。默认为None。详情请参考 :ref:`api_guide_Name`。

变量：
    - **weight_ih** (Parameter) - input到hidden的变换矩阵的权重。形状为（4 * hidden_size, input_size）。对应公式中的 :math:`W_{ii}, W_{if}, W_{ig}, W_{io}`。
    - **weight_hh** (Parameter) - hidden到hidden的变换矩阵的权重。形状为（4 * hidden_size, hidden_size）。对应公式中的 :math:`W_{hi}, W_{hf}, W_{hg}, W_{ho}`。
    - **bias_ih** (Parameter) - input到hidden的变换矩阵的偏置。形状为（4 * hidden_size, ）。对应公式中的 :math:`b_{ii}, b_{if}, b_{ig}, b_{io}`。
    - **bias_hh** (Parameter) - hidden到hidden的变换矩阵的偏置。形状为（4 * hidden_size, ）。对应公式中的 :math:`b_{hi}, b_{hf}, b_{hg}, b_{ho}`。
    
输入:
    - **inputs** (Tensor) - 输入。形状为[batch_size, input_size]，对应公式中的 :math:`x_t`。
    - **states** (tuple，可选) - 一个包含两个Tensor的元组，每个Tensor的形状都为[batch_size, hidden_size]，上一轮的隐藏状态。对应公式中的 :math:`h_{t-1}，c_{t-1}`。当state为None的时候，初始状态为全0矩阵。默认为None。

输出:
    - **outputs** (Tensor) - 输出。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。
    - **new_states** (tuple) - 一个包含两个Tensor的元组，每个Tensor的形状都为[batch_size, hidden_size]，新一轮的隐藏状态。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}，c_{t}`。
    
.. Note::
    所有的变换矩阵的权重和偏置都默认初始化为Uniform(-std, std)，其中std = :math:`\frac{1}{\sqrt{hidden_size}}`。对于参数初始化，详情请参考 :ref:`api_fluid_ParamAttr`。


**代码示例**：

.. code-block:: python

            import paddle

            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))
            prev_c = paddle.randn((4, 32))

            cell = paddle.nn.LSTMCell(16, 32)
            y, (h, c) = cell(x, (prev_h, prev_c))
            print(y.shape)
            print(h.shape)
            print(c.shape)
            
            #[4,32]
            #[4,32]
            #[4,32]
