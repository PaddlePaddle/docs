.. _cn_api_amp_debugging_check_numerics:

check_numerics
-------------------------------

.. py:function:: paddle.amp.debugging.check_numerics(tensor, op_type, var_name, debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF_AND_ABORT)

用来调试一个 Tensor ，计算并返回这个 Tensor 数值中异常值(NaN、Inf) 和零元素的数量。


参数
:::::::::

- **tensor** (Tensor) – 需要检查的目标 Tensor 。
- **op_type** (str) – 产生目标 Tensor 的 OP 或 API 。
- **var_name** (str) – 目标 Tensor 的名字。
- **debug_mode** (paddle.amp.debugging.DebugMode, 可选) - 要使用的调试类型。默认值为 ``paddle.amp.debugging.DebugMode.CHECK_NAN_INF_AND_ABORT``。

返回
:::::::::

- stats(Tensor)，保存目标 Tensor 统计信息的 Tensor ，形状为 [3]，依次存放目标 Tensor 中 NaN、Inf 和零元素的数量。数据类型为 int64。
- values(Tensor)，保存目标 Tensor 的最大值、最小值和所有元素的均值，形状为 [3]，数据类型为 float 。

代码示例
::::::::::::

COPY-FROM: paddle.amp.debugging.check_numerics
