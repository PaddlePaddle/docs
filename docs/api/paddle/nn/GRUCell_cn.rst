.. _cn_api_paddle_nn_GRUCell:

GRUCell
-------------------------------

.. py:class:: paddle.nn.GRUCell(input_size, hidden_size, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)



**门控循环单元**

该 OP 是门控循环单元（GRUCell），根据当前时刻输入 x（t）和上一时刻状态 h（t-1）计算当前时刻输出 y（t）并更新状态 h（t）。

状态更新公式如下：

..  math::

        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}h_{t-1} + b_{hr})

        z_{t} & = \sigma(W_{iz}x_{t} + b_{iz} + W_{hz}h_{t-1} + b_{hz})

        \widetilde{h}_{t} & = \tanh(W_{ic}x_{t} + b_{ic} + r_{t} * (W_{hc}h_{t-1} + b_{hc}))

        h_{t} & = z_{t} * h_{t-1} + (1 - z_{t}) * \widetilde{h}_{t}

        y_{t} & = h_{t}

其中：

    - :math:`\sigma` ：sigmoid 激活函数。

详情请参考论文：`An Empirical Exploration of Recurrent Network Architectures <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ 。


参数
::::::::::::

    - **input_size** (int) - 输入的大小。
    - **hidden_size** (int) - 隐藏状态大小。
    - **weight_ih_attr** (ParamAttr，可选) - weight_ih 的参数。默认为 None。
    - **weight_hh_attr** (ParamAttr，可选) - weight_hh 的参数。默认为 None。
    - **bias_ih_attr** (ParamAttr，可选) - bias_ih 的参数。默认为 None。
    - **bias_hh_attr** (ParamAttr，可选) - bias_hh 的参数。默认为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

变量
::::::::::::

    - **weight_ih** (Parameter) - input 到 hidden 的变换矩阵的权重。形状为（3 * hidden_size, input_size）。对应公式中的 :math:`W_{ir}, W_{iz}, W_{ic}`。
    - **weight_hh** (Parameter) - hidden 到 hidden 的变换矩阵的权重。形状为（3 * hidden_size, hidden_size）。对应公式中的 :math:`W_{hr}, W_{hz}, W_{hc}`。
    - **bias_ih** (Parameter) - input 到 hidden 的变换矩阵的偏置。形状为（3 * hidden_size, ）。对应公式中的 :math:`b_{ir}, b_{iz}, b_{ic}`。
    - **bias_hh** (Parameter) - hidden 到 hidden 的变换矩阵的偏置。形状为（3 * hidden_size, ）。对应公式中的 :math:`b_{hr}, b_{hz}, b_{hc}`。

输入
::::::::::::

    - **inputs** (Tensor) - 输入。形状为[batch_size, input_size]，对应公式中的 :math:`x_t`。
    - **states** (Tensor，可选) - 上一轮的隐藏状态。对应公式中的 :math:`h_{t-1}`。当 state 为 None 的时候，初始状态为全 0 矩阵。默认为 None。

输出：
    - **outputs** (Tensor) - 输出。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。
    - **new_states** (Tensor) - 新一轮的隐藏状态。形状为[batch_size, hidden_size]，对应公式中的 :math:`h_{t}`。

.. note::
    所有的变换矩阵的权重和偏置都默认初始化为 Uniform(-std, std)，其中 std = :math:`\frac{1}{\sqrt{hidden\_size}}`。对于参数初始化，详情请参考 :ref:`cn_api_paddle_ParamAttr`。


代码示例
::::::::::::

COPY-FROM: paddle.nn.GRUCell
