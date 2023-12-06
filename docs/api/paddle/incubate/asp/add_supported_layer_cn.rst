.. _cn_api_paddle_incubate_asp_add_supported_layer:

add_supported_layer
-------------------------------

.. py:function:: paddle.incubate.asp.add_supported_layer(layer, pruning_func=None)

添加支持的 layer 及其相应的剪枝函数。

参数
:::::::::
    - **name** (string|Layer) - 需要支持的 layer 的名称或类型。 如果 layer 是 Layer，那么它将在内部转为字符串。ASP 将使用这个名称来匹配参数的名称并调用其相应的剪枝函数。
    - **pruning_func** (function，可选) - 一个接收五个参数（weight_nparray、m、n、func_name、param_name）的函数类型，weight_nparray 是 nparray 格式的权重，param_name 是权重的名称，m、n 和 func_name，详细信息请参见 :ref:`cn_api_paddle_incubate_asp_prune_model`。
