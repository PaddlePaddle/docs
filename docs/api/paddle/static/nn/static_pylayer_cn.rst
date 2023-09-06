.. _cn_api_fluid_layers_static_pylayer:

static_pylayer
-------------------------------


.. py:function:: paddle.static.nn.static_pylayer(forward_fn, inputs, backward_fn=None, name=None)


该 API 返回 ``forward_fn(inputs)``，并且根据传入的 ``forward_fn`` 和 ``backward_fn`` 的执行逻辑创建两个 sub_block，
同时创建 ``pylayer`` 算子，``pylayer`` 算子的属性储存创建的 sub_block ID。

``forward_fn`` 和 ``backward_fn`` 需要返回同样嵌套结构（nest structure）的 Tensor。
PaddlePaddle 里 Tensor 的嵌套结构是指一个 Tensor，或者 Tensor 的元组（tuple），或者 Tensor 的列表（list）。

.. note::
    1. 如果 ``backward_fn`` 被设置为 None，用户需要使 ``forward_fn`` 的输入数量和 ``backward_fn`` 的输出数量相同，``forward_fn`` 的输出数量和 ``backward_fn`` 的输入数量相同。
    2. 在 ``backward_fn`` 被设置为 ``None`` 的情况下，``inputs`` 里所有 Variable 的 ``stop_gradient`` 属性应该被设为 ``True``，否则可能会在反向传播（backward propagation）中得到意想不到的结果。
    3. 本 API 只能被运行在静态图模式下。

参数
:::::::::
    - **forward_fn** (callable) - 一个前向传播（forward propagation）时被调用的 callable。
    - **inputs** (list[Variable]) - Variable 类型列表，其含义为 ``forward_fn`` 的输入 Variable。
    - **backward_fn** (callable，可选) - 一个反向传播（backward propagation）时被调用的 callable。默认值：``None``，表示不需要进行反向传播（backward propagation）。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值：``None``。

返回
:::::::::
Variable|list(Variable)|tuple(Variable)，该 API 返回 ``forward_fn(inputs)``。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.static_pylayer
