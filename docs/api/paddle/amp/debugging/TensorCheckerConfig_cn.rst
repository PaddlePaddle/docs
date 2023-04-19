.. _cn_api_debugging_debugmode:

debugmode
-------------------------------

.. py:function:: paddle.amp.debugging.DebugMode
用于设置检查模块或运算张量中的 nan 和 inf 的配置。
    参数：
    - enable：是否开启精度检查功能。默认值为 False，表示永远不会使用这些工具。
    - debug_mode：调试模式，有6种调试模式。
        CHECK_NAN_INF_AND_ABORT(default)：用NaN/Inf打印或保存Tensor key信息，中断程序
        CHECK_NAN_INF：用NaN/Inf打印或保存Tensor关键信息，但继续运行
        CHECK_ALL_FOR_OVERFLOW：检查FP32算子的输出，打印或保存超出FP16表示范围的关键Tensor信息（上溢、下溢）
        CHECK_ALL：打印或保存所有算子的输出Tensor关键信息
    - output_dir：精度检查日志存放路径。如果为None，则直接打印到终端
    - checked_op_list：要检查的算子列表
    - skipped_op_list: 跳过检查的算子列表
    - debug_step: 调试的迭代范围, 用于控制模型迭代
    - stack_height_limit: 调用栈的最大深度，支持打印错误位置的调用栈。
    - enable_traceback_filtering：是否过滤traceback。主要目的是过滤掉框架内部代码调用栈，只显示用户代码调用栈

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

