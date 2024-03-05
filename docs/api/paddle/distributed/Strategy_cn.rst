.. _cn_api_paddle_distributed_Strategy:

Strategy
-------------------------------

.. py:class:: paddle.distributed.Strategy

用于配置静态图分布式训练时所使用的并行策略和优化策略。目前支持 ``sharding``、 ``fused_passes``、 ``gradient_merge`` 和 ``pipeline``，未来将支持更多的策略。

``sharding`` 用于配置优化器的分片策略，可以节省 GPU 显存。

``fused_passes`` 用于配置模型的计算融合策略。

``gradient_merge`` 用于配置训练时的梯度融合。

``pipeline`` 用于配置流水线并行策略。


参数
:::::::::

    - **config** (dict|None, 可选) - 策略配置。若 ``config`` 参数为 ``None``，则使用默认配置。若 ``config`` 为字典，则将字典中包含的项设置为用户自定义的配置，字典中不包含的项依旧使用默认配置。默认：None。


**代码示例**

COPY-FROM: paddle.distributed.Strategy


属性
::::::::::::

sharding
'''''''''

优化器分片策略，包含以下配置项：

    - **``enable``** (bool) - 是否启用优化器分片策略。默认：False。

    - **``stage``** (int) - 可以设置为 1、2 或 3。1 表示切分优化器状态，2 代表切分优化器状态和梯度，3 表示切分优化器状态、梯度和参数。默认：1。

    - **``degree``** (int) - 分片的数量。默认：8。

**代码示例**

COPY-FROM: paddle.distributed.Strategy.sharding


fused_passes
'''''''''''''

计算融合策略，包含以下配置项：

    - **``enable``** (bool) - 是否启用计算融合策略。默认：False。

    - **``gemm_epilogue``** (bool) - 是否融合 ``Linear`` 层中的 ``matmul`` 和 ``add`` 计算。默认：False。

    - **``dropout_add``** (bool) - 是否融合 ``dropout`` 和 ``add`` 计算。默认：False。

**代码示例**

COPY-FROM: paddle.distributed.Strategy.fused_passes


gradient_merge
'''''''''''''''

梯度融合策略，包含以下配置项：

    - **``enable``** (bool) - 是否启用梯度融合策略。默认：False。

    - **``k_steps``** (int) - 梯度融合的步数。默认：1。

    - **``avg``** (bool) - 是否平均梯度。默认：True。

**代码示例**

COPY-FROM: paddle.distributed.Strategy.gradient_merge


pipeline
'''''''''

流水线并行策略，包含以下配置项：

    - **``enable``** (bool) - 是否启用流水线并行策略。默认：False。

    - **``schedule_mode``** (str) - 流水线并行的调度模式。默认：1F1B。

    - **``micro_batch_size``** (int) - mini-batch 中包含的每个 micro-batch 的大小。默认：1。

    - **``accumulate_steps``** (int) - 累积步数。默认：1。

**代码示例**

COPY-FROM: paddle.distributed.Strategy.pipeline
