.. _cn_api_debugging_debugmode:

debugmode
-------------------------------

.. py:function:: paddle.amp.debugging.DebugMode

DebugMode 用于标识 TensorCheckerConfig 的状态。
    每个DebugMode的含义如下
    - **DebugMode.CHECK_NAN_INF_AND_ABORT** : 打印或保存带有NaN/Inf的Tensor关键信息并中断程序
    - **DebugMode.CHECK_NAN_INF** : 打印或保存带有 NaN/Inf的Tensor 关键信息，但继续运行
    - **DebugMode.CHECK_ALL_FOR_OVERFLOW** : 检查FP32算子的输出，打印或保存超过FP16表示范围的关键Tensor信息（上溢、下溢）
    - **DebugMode.CHECK_ALL** : 打印或保存所有算子的输出Tensor关键信息

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

