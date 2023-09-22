.. _cn_api_paddle_amp_debugging_enable_operator_stats_collection:

enable_operator_stats_collection
-------------------------------

.. py:function:: paddle.amp.debugging.enable_operator_stats_collection()

启用以收集不同数据类型的算子调用次数。按照 float32、float16、bfloat16 四种数据类型统计算子调用次数, 此函数与相应的禁用函数配对使用。

返回
:::::::::
无返回值

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.enable_operator_stats_collection
