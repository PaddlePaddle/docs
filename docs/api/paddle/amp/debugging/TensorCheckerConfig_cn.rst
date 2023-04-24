.. _cn_api_amp_debugging_tensor_checker_config:

TensorCheckerConfig
-------------------------------

.. py:class:: paddle.amp.debugging.TensorCheckerConfig(enable, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT, output_dir=None, checked_op_list=None, skipped_op_list=None, debug_step=None, stack_height_limit=3)

该函数的目的是收集用于检查模块或运算符张量中 NaN 和 Inf 值的配置。

参数
:::::::::
    - **enable** : 布尔值，指示是否启用检测张量中的 NaN 和 Inf 值。默认值为 False，这意味着不使用这些工具。
    - **debug_mode** : 确定要使用的调试类型的参数。有 4 种可用模式：
        - **CHECK_NAN_INF_AND_ABORT** (默认）: 此模式打印或保存包含 NaN/Inf 的张量的关键信息，并中断程序。
        - **CHECK_NAN_INF** : 此模式打印或保存包含 NaN/Inf 的张量的关键信息，但允许程序继续运行。
        - **CHECK_ALL_FOR_OVERFLOW** : 此模式检查 FP32 运算符的输出，并打印或保存超出 FP16 表示范围的关键张量的信息，例如溢出或下溢。
        - **CHECK_ALL** : 此模式打印或保存所有运算符的输出张量关键信息。
    - **output_dir** : 存储收集数据的路径。如果将此参数设置为None，则数据将打印到终端。
    - **checked_op_list** : 指定程序运行过程中需要检查的算子列表，例如checked_op_list=['elementwise_add‘，’conv2d']，表示程序运行过程中对elementwise_add和conv2d的输出结果进行nan/inf检查。
    - **skipped_op_list** : 指定程序运行过程中不需要检查的算子列表，例如skipped_op_list=['elementwise_add‘，’conv2d']，表示程序运行过程中不对elementwise_add和conv2d的输出结果进行nan/inf检查。
    - **debug_step** : 列表或元组，主要用于模型训练过程中的nan/inf检查。例如：debug_step=[1,5]表示只对模型训练迭代1~5之间进行nan/inf检查。
    - **stack_height_limit** : 整数值，指定调用栈的最大深度。该功能支持在错误位置打印调用栈。目前仅支持开启或关闭调用栈打印。如果您想在GPU Kernel检测到NaN时打印相应的C++调用栈，可以将stack_height_limit设置为1，否则设置为0。

代码示例
:::::::::

COPY-FROM: paddle.amp.debugging.TensorCheckerConfig
