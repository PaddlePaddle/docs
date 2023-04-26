.. _cn_api_amp_debugging_collect_operator_stats:

collect_operator_stats
-------------------------------

.. py:function:: paddle.amp.debugging.collect_operator_stats()

上下文切换器能够收集不同数据类型的运算符数量。按照 float32、float16、bfloat16 等四种数据类型进行分类统计数据算子调用次数，在退出 context 时打印。

返回
:::::::::
无返回值

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.collect_operator_stats
