.. _cn_api_amp_debugging_tensor_checker_config:

TensorCheckerConfig
-------------------------------

.. py:class:: paddle.amp.debugging.TensorCheckerConfig(enable, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT, output_dir=None, checked_op_list=None, skipped_op_list=None, debug_step=None, stack_height_limit=1)

该函数的目的是收集用于检查模块或运算符张量中 NaN 和 Inf 值的配置。

参数
:::::::::
    - **enable（bool)** : 布尔值，指示是否启用检测张量中的 NaN 和 Inf 值。默认值为 False，这意味着不使用这些工具。
    - **debug_mode(DebugMode, 可选)** : 确定要使用的调试类型的参数。默认 CHECK_NAN_INF_AND_ABORT，此模式打印或保存包含 NaN/Inf 的张量的关键信息，并中断程序。
    - **output_dir(str, 可选）** : 用于存储检查日志的文件路径，默认为 None。
    - **checked_op_list(list|tuple, 可选)** : 指定程序运行过程中需要检查的算子列表，例如 checked_op_list=['elementwise_add', 'conv2d']，表示程序运行过程中对 elementwise_add 和 conv2d 的输出结果进行 NaN/Inf 检查。默认为 None。
    - **skipped_op_list(list|tuple, 可选)** : 指定程序运行过程中不需要检查的算子列表，例如 skipped_op_list=['elementwise_add', 'conv2d']，表示程序运行过程中不对 elementwise_add 和 conv2d 的输出结果进行 NaN/Inf 检查。默认为 None。
    - **debug_step(list|tuple, 可选)** : 列表或元组，主要用于模型训练过程中的 NaN/Inf 检查。例如： debug_step=[1, 5] 表示只对模型训练迭代 1~5 之间进行 NaN/Inf 检查。默认为 None。
    - **stack_height_limit(int, 可选)** : 整数值，指定调用栈的最大深度。该功能支持在错误位置打印调用栈。目前仅支持开启或关闭调用栈打印。如果您想在 GPU Kernel 检测到 NaN 时打印相应的 C++ 调用栈，可以将 stack_height_limit 设置为 1，否则设置为 0。默认为 1。

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.TensorCheckerConfig
