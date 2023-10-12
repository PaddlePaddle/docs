.. _cn_api_paddle_incubate_asp_set_excluded_layers:

set_excluded_layers
-------------------------------

.. py:function:: paddle.incubate.asp.set_excluded_layers(param_names, main_program=None)


设置不会被裁剪为稀疏权重的 layer 的参数名称。

参数
:::::::::

    - **param_names** (list of string) - 包含参数名的列表
    - **main_program** (Program，可选) - 包含模型定义及其参数的 Program。如果为 None，那么它将被设置为 paddle.static.default_main_program()。默认为 None。

代码示例
::::::::::::

1. 动态图模式

COPY-FROM: paddle.incubate.asp.set_excluded_layers:dynamic-graph

2. 静态图模式

COPY-FROM: paddle.incubate.asp.set_excluded_layers:static-graph
