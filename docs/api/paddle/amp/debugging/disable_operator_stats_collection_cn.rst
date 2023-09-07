.. _cn_api_paddle_amp_debugging_disable_operator_stats_collection:

disable_operator_stats_collection
-------------------------------

.. py:function:: paddle.amp.debugging.disable_operator_stats_collection()

禁用收集不同数据类型的算子调用次数。该函数与相应的使能函数配对使用。按照 float32、float16、bfloat16 等四种数据类型进行分类统计算子调用次数。

返回
:::::::::
无返回值

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.disable_operator_stats_collection
