.. _cn_api_amp_debugging_enable_tensor_checker(checker_config)

paddle.amp.debugging.enable_tensor_checker(checker_config)
-------------------------------

.. py:function:: paddle.amp.debugging.enable_tensor_checker()

enable_tensor_checker(checker_config)是开启模型级别的精度检查，与disables_tensor_checker()一起使用，通过这两个API的组合实现模型级别的精度检查，检查指定范围内所有算子的输出Tensor。
    注意：
    * 如果backward()之前调用disable_tensor_checker(), 则不检查梯度算子；
    * 如果在optimizer.step()之前调用了disable_tensor_checker(), 则不会检查优化器和其他权重更新相关的算子

**代码示例**：
.. code-block:: python

    import paddle

    checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF)
    paddle.amp.debugging.enable_tensor_checker(checker_config)
    x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
    y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
    res = paddle.pow(x, y)
    paddle.autograd.backward(res, retain_graph=True)
    paddle.amp.debugging.disable_tensor_checker()

    #[PRECISION] [ERROR] in [device=cpu, op=elementwise_pow_grad, tensor=, dtype=fp32], numel=3, num_nan=1, num_inf=0, num_zero=0, max=2.886751e-01, min=2.000000e-01, mean=-nan
