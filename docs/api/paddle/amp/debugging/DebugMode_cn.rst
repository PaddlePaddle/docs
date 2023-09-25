.. _cn_api_paddle_amp_debugging_DebugMode:

DebugMode
-------------------------------

.. py:class:: paddle.amp.debugging.DebugMode()

`DebugMode` 用于标识 `TensorCheckerConfig` 的状态。每个 `DebugMode` 的含义如下:

    - **DebugMode.CHECK_NAN_INF_AND_ABORT** - 打印或保存带有 NaN/Inf 的 Tensor 关键信息并中断程序。

    - **DebugMode.CHECK_NAN_INF** - 打印或保存带有 NaN/Inf 的 Tensor 关键信息，但继续运行程序。

    - **DebugMode.CHECK_ALL_FOR_OVERFLOW** - 检查 FP32 算子的输出，打印或保存超过 FP16 表示范围的关键 Tensor 信息（上溢、下溢）。

    - **DebugMode.CHECK_ALL** - 打印或保存所有算子的输出 Tensor 关键信息。
