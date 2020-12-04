.. _cn_api_paddle_nn_layer_rnn_SimpleRNNCell:

SimpleRNNCell
-------------------------------

.. py:class:: paddle.nn.SimpleRNNCell(input_size, hidden_size, activation="tanh", weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)



**简单循环神经网络单元**

该OP是简单循环神经网络单元（SimpleRNNCell），根据当前时刻输入x（t）和上一时刻状态h（t-1）计算当前时刻输出y（t）并更新状态h（t）。

状态更新公式如下：

.. math::

        h_{t} & = \mathrm{act}(W_{ih}x_{t} + b_{ih} + W_{hh}h_{t-1} + b_{hh})

        y_{t} & = h_{t}

其中的 `act` 表示激活函数。

详情请参考论文 :`Finding Structure in Time <https://crl.ucsd.edu/~elman/Papers/fsit.pdf>`_。

参数：
    - **input_size** (int) - 输入的大小。
    - **hidden_size** (int) - 隐藏状态大小。
    - **activation** (str, 可选) - 简单循环神经网络单元的激活函数。可以是tanh或relu。默认为tanh。
    - **weight_ih_attr** (ParamAttr，可选) - weight_ih的参数。默认为None。
    - **weight_hh_attr** (ParamAttr，可选) - weight_hh的参数。默认为None。
    - **bias_ih_attr** (ParamAttr，可选) - bias_ih的参数。默认为None。
    - **bias_hh_attr** (ParamAttr，可选) - bias_hh的参数。默认为None。
    - **name** (str, 可选): OP的名字。默认为None。详情请参考 :ref:`api_guide_Name`。

变量：
    - **weight_ih** (Parameter) - input到hidden的变换矩阵的权重。形状为（hidden_size, input_size）。对应公式中的 :math:`W_{ih}`。
    - **weight_hh** (Parameter) - hidden到hidden的变换矩阵的权重。形状为（hidden_size, hidden_size）。对应公式中的 :math:`W_{hh}`。
    - **bias_ih** (Parameter) - input到hidden的变换矩阵的偏置。形状为（hidden_size, ）。对应公式中的 :math:`b_{ih}`。
    - **bias_hh** (Parameter) - hidden到hidden的变换矩阵的偏置。形状为（hidden_size, ）。对应公式中的 :math:`b_{hh}`。
    
输入:
    - **inputs** (Tensor) - 输入。形状为[batch_size, input_size]，对应公式中的 :math:`x_t`。
    - **states** (Tensor，可选) - 上一轮的隐藏状态。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t-1}`。当state为None的时候，初始状态为全0矩阵。默认为None。

输出:
    - **outputs** (Tensor) - 输出。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。
    - **new_states** (Tensor) - 新一轮的隐藏状态。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。
    
.. Note::
    所有的变换矩阵的权重和偏置都默认初始化为Uniform(-std, std)，其中std = :math:`\frac{1}{\sqrt{hidden\_size}}`。对于参数初始化，详情请参考 :ref:`api_fluid_ParamAttr`。


**代码示例**：

.. code-block:: python

            import paddle

            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.SimpleRNNCell(16, 32)
            y, h = cell(x, prev_h)
            print(y.shape)
            
            #[4,32]
