.. _cn_api_paddle_amp_debugging_disable_tensor_checker:

disable_tensor_checker
-------------------------------

.. py:function:: paddle.amp.debugging.disable_tensor_checker()

disable_tensor_checker() 用于禁用精度检查，通常与 enable_tensor_checker(config) 一起使用，通过检查指定范围内所有算子的输出张量来实现模型级别的精度检查。

返回
:::::::::
无返回值

.. note::
    如果在 backward() 之前调用 disable_tensor_checker()，则不会检查梯度算子；
    如果在 optimizer.step() 之前调用 disable_tensor_checker()，则不会检查优化器和其他与权重更新相关的算子。

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.disable_tensor_checker
