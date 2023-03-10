.. _cn_api_paddle_nn_layer_rnn_SimpleRNNCell:

SimpleRNNCell
-------------------------------

.. py:class:: paddle.nn.SimpleRNNCell(input_size, hidden_size, activation="tanh", weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)



**简单循环神经网络单元**

简单循环神经网络单元（SimpleRNNCell），根据当前时刻输入 x（t）和上一时刻状态 h（t-1）计算当前时刻输出 y（t）并更新状态 h（t）。

状态更新公式如下：

.. math::

        h_{t} & = \mathrm{act}(W_{ih}x_{t} + b_{ih} + W_{hh}h_{t-1} + b_{hh})

        y_{t} & = h_{t}

其中的 `act` 表示激活函数。

详情请参考论文：`Finding Structure in Time <https://onlinelibrary.wiley.com/doi/pdf/10.1207/s15516709cog1402_1>`_ 。

参数
::::::::::::

    - **input_size** (int) - 输入的大小。
    - **hidden_size** (int) - 隐藏状态大小。
    - **activation** (str，可选) - 简单循环神经网络单元的激活函数。可以是 tanh 或 relu。默认为 tanh。
    - **weight_ih_attr** (ParamAttr，可选) - weight_ih 的参数。默认为 None。
    - **weight_hh_attr** (ParamAttr，可选) - weight_hh 的参数。默认为 None。
    - **bias_ih_attr** (ParamAttr，可选) - bias_ih 的参数。默认为 None。
    - **bias_hh_attr** (ParamAttr，可选) - bias_hh 的参数。默认为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

变量
::::::::::::

    - **weight_ih** (Parameter) - input 到 hidden 的变换矩阵的权重。形状为（hidden_size, input_size）。对应公式中的 :math:`W_{ih}`。
    - **weight_hh** (Parameter) - hidden 到 hidden 的变换矩阵的权重。形状为（hidden_size, hidden_size）。对应公式中的 :math:`W_{hh}`。
    - **bias_ih** (Parameter) - input 到 hidden 的变换矩阵的偏置。形状为（hidden_size, ）。对应公式中的 :math:`b_{ih}`。
    - **bias_hh** (Parameter) - hidden 到 hidden 的变换矩阵的偏置。形状为（hidden_size, ）。对应公式中的 :math:`b_{hh}`。

输入
::::::::::::

    - **inputs** (Tensor) - 输入。形状为[batch_size, input_size]，对应公式中的 :math:`x_t`。
    - **states** (Tensor，可选) - 上一轮的隐藏状态。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t-1}`。当 state 为 None 的时候，初始状态为全 0 矩阵。默认为 None。

输出
::::::::::::

    - **outputs** (Tensor) - 输出。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。
    - **new_states** (Tensor) - 新一轮的隐藏状态。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。

.. note::
    所有的变换矩阵的权重和偏置都默认初始化为 Uniform(-std, std)，其中 std = :math:`\frac{1}{\sqrt{hidden\_size}}`。对于参数初始化，详情请参考 :ref:`cn_api_fluid_ParamAttr`。

代码示例
::::::::::::

COPY-FROM: paddle.nn.SimpleRNNCell
