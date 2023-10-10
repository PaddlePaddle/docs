.. _cn_api_paddle_incubate_asp_reset_excluded_layers:

reset_excluded_layers
-------------------------------

.. py:function:: paddle.incubate.asp.reset_excluded_layers(main_program=None)


重置与 main_program 对应的 excluded_layers 设置。如果 main_program 为 None，则 excepted_layers 的所有配置都将被清除。


参数
::::::::::::

- **main_program** (Program，可选) - 包含模型定义及其参数的 Program。如果给出 None，那么这个函数将重置所有 exclusion_layers。 默认为 None。

代码示例
::::::::::::

1. 动态图模式

COPY-FROM: paddle.incubate.asp.reset_excluded_layers:dynamic_graph

2. 静态图模式

COPY-FROM: paddle.incubate.asp.reset_excluded_layers:static_graph
