.. _cn_api_amp_debugging_enable_tensor_checker

enable_tensor_checker
-------------------------------

.. py:function:: paddle.amp.debugging.enable_tensor_checker(checker_config)

enable_tensor_checker(checker_config)是开启模型级别的精度检查，与disables_tensor_checker()一起使用，通过这两个API的组合实现模型级别的精度检查，检查指定范围内所有算子的输出Tensor。

参数
:::::::::
    - **checker_config**: TensorCheckerConfig类型的参数，用于设置精度检查工具的调试选项。

返回
:::::::::
无返回值

.. note:: 
    如果在backward()之前调用disable_tensor_checker()，则不会检查梯度算子；
    如果在optimizer.step()之前调用disable_tensor_checker()，则不会检查优化器和其他与权重更新相关的算子。

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.enable_tensor_checker
