.. _cn_api_distributed_fleet_DistributedStrategy:

DistributedStrategy
-------------------------------

.. py:class:: paddle.distributed.fleet.DistributedStrategy



属性
::::::::::::

save_to_prototxt
'''''''''

序列化当前的 DistributedStrategy，并且保存到 output 文件中

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.dgc = True
  strategy.recompute = True
  strategy.recompute_configs = {"checkpoints": ["x"]}
  strategy.save_to_prototxt("dist_strategy.prototxt")


load_from_prototxt
'''''''''

加载已经序列化过的 DistributedStrategy 文件，并作为初始化 DistributedStrategy 返回

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.load_from_prototxt("dist_strategy.prototxt")


execution_strategy
'''''''''

`Post Local SGD <https://arxiv.org/abs/1808.07217>`__

配置 DistributedStrategy 中的 `ExecutionStrategy <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/compiler/ExecutionStrategy_cn.html>`_

**代码示例**

.. code-block:: python

  import paddle
  exe_strategy = paddle.static.ExecutionStrategy()
  exe_strategy.num_threads = 10
  exe_strategy.num_iteration_per_drop_scope = 10
  exe_strategy.num_iteration_per_run = 10

  strategy = paddle.distributed.fleet.DistributedStrategy()
  strategy.execution_strategy = exe_strategy


build_strategy
'''''''''

配置 DistributedStrategy 中的 `BuildStrategy <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/compiler/BuildStrategy_cn.html>`_

**代码示例**

.. code-block:: python

  import paddle
  build_strategy = paddle.static.BuildStrategy()
  build_strategy.enable_sequential_execution = True
  build_strategy.fuse_elewise_add_act_ops = True
  build_strategy.fuse_bn_act_ops = True
  build_strategy.enable_auto_fusion = True
  build_strategy.fuse_relu_depthwise_conv = True
  build_strategy.fuse_broadcast_ops = True
  build_strategy.fuse_all_optimizer_ops = True
  build_strategy.enable_inplace = True

  strategy = paddle.distributed.fleet.DistributedStrategy()
  strategy.build_strategy = build_strategy


auto
'''''''''

表示是否启用自动并行策略。此功能目前是实验性功能。目前，自动并行只有在用户只设置 auto，不设置其它策略时才能生效。具体请参考示例代码。默认值：False

**代码示例**

.. code-block:: python

  import paddle
  import paddle.distributed.fleet as fleet
  paddle.enable_static()

  strategy = fleet.DistributedStrategy()
  strategy.auto = True
  # if set other strategy at the same time, auto will not apply
  # strategy.amp = True

  optimizer = paddle.optimizer.SGD(learning_rate=0.01)
  optimizer = fleet.distributed_optimizer(optimizer, strategy)


recompute
'''''''''

是否启用 Recompute 来优化内存空间，默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.recompute = True
  # suppose x and y are names of checkpoint tensors for recomputation
  strategy.recompute_configs = {
    "checkpoints": ["x", "y"],
    "enable_offload": True,
    "checkpoint_shape": [100, 512, 1024]
    }


recompute_configs
'''''''''

设置 Recompute 策略的配置。目前来讲，用户使用 Recompute 策略时，必须配置 checkpoints 参数。

**checkpoints(int):** Recompute 策略的检查点，默认为空列表，也即不启用 Recompute。

**enable_offload(bool):** 是否开启 recompute-offload 策略。该策略会在 recompute 的基础上，将原本驻留在显存中的 checkpoints 卸载到 Host 端的内存中，进一步更大的 batch size。因为 checkpoint 在内存和显存间的拷贝较慢，该策略是通过牺牲速度换取更大的 batch size。默认值：False。

**checkpoint_shape(list):** 该参数仅在 offload 开启时需要设置，用来指定 checkpoints 的各维度大小。目前 offload 需要所有 checkpoints 具有相同的 shape，并且各维度是确定的（不支持 -1 维度）。


pipeline
'''''''''

是否启用 Pipeline 并行。目前，主要实现单机多 GPU 间的 Pipeline 并行和多机间的数据并行。Pipeline 信息由用户定义程序中的 device_guard 确定。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.pipeline = True


pipeline_configs
'''''''''

设置 Pipeline 策略的配置。Pipeline 策略下，神经网络的不同层在不同的 GPU 设备。相邻的 GPU 设备间有用于同步隐层 Tensor 的队列。Pipeline 并行包含多种生产者-消费者形式的硬件对，如 GPU-CPU、CPU-GPU、GPU-XPU。加速 PIpeline 并行的最佳方式是减少 Tensor 队列中的 Tensor 大小，这样生产者可以更快的为下游消费者提供数据。

**micro_batch_size (int):** 每个用户定义的 mini-batch 中包含的更小的 micro-batch 的数量。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.pipeline = True
  strategy.pipeline_configs = {"micro_batch_size": 12}


gradient_merge
'''''''''

梯度累加，是一种大 Batch 训练的策略。添加这一策略后，模型的参数每过 **k_steps** 步更新一次，
**k_steps** 是用户定义的步数。在不更新参数的步数里，Paddle 只进行前向、反向网络的计算；
在更新参数的步数里，Paddle 执行优化网络，通过特定的优化器（比如 SGD、Adam），
将累加的梯度应用到模型参数上。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.gradient_merge = True
  strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}

gradient_merge_configs
'''''''''

设置 **distribute_strategy** 策略的配置。

**k_steps(int):** 参数更新的周期，默认为 1

**avg(bool):** 梯度的融合方式，有两种选择：

- **sum**：梯度求和
- **avg**：梯度求平均


lars
'''''''''

是否使用 LARS optimizer，默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.lars = True
  strategy.lars_configs = {
    "lars_coeff": 0.001,
    "lars_weight_decay": 0.0005,
    "epsilon": 0,
    "exclude_from_weight_decay": ["batch_norm", ".b"],
  }

lars_configs
'''''''''

设置 LARS 优化器的参数。用户可以配置 lars_coeff，lars_weight_decay，epsilon，exclude_from_weight_decay 参数。

**lars_coeff(float):** lars 系数，`原论文 <https://arxiv.org/abs/1708.03888>`_ 中的 trust coefficient。默认值是 0.001。

**lars_weight_decay(float):** lars 公式中 weight decay 系数。默认值是 0.0005。

**exclude_from_weight_decay(list[str]):** 不应用 weight decay 的 layers 的名字列表，某一 layer 的 name 如果在列表中，这一 layer 的 lars_weight_decay 将被置为 0。默认值是 None。

**epsilon(float):** 一个小的浮点值，目的是维持数值稳定性，避免 lars 公式中的分母为零。默认值是 0。


lamb
'''''''''

是否使用 LAMB optimizer，默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.lamb = True
  strategy.lamb_configs = {
      'lamb_weight_decay': 0.01,
      'exclude_from_weight_decay': [],
  }

lamb_configs
'''''''''

设置 LAMB 优化器的参数。用户可以配置 lamb_weight_decay，exclude_from_weight_decay 参数。

**lamb_weight_decay(float):** lars 公式中 weight decay 系数。默认值是 0.01。

**exclude_from_weight_decay(list[str]):** 不应用 weight decay 的 layers 的名字列表，某一 layer 的 name 如果在列表中，这一 layer 的 lamb_weight_decay 将被置为 0。默认值是 None。


localsgd
'''''''''
是否使用 LocalSGD optimizer，默认值：False。更多的细节请参考 `Don't Use Large Mini-Batches, Use Local SGD <https://arxiv.org/pdf/1808.07217.pdf>`_

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.localsgd = True # by default this is false


localsgd_configs
'''''''''
设置 LocalSGD 优化器的参数。用户可以配置 k_steps 和 begin_step 参数。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.localsgd = True
  strategy.localsgd_configs = {"k_steps": 4,
                                "begin_step": 30}

**k_steps(int):** 训练过程中的全局参数更新间隔，默认值 1。

**begin_step(int):** 指定从第几个 step 之后进行 local SGD 算法，默认值 1。

adaptive_localsgd
'''''''''
是否使用 AdaptiveLocalSGD optimizer，默认值：False。更多的细节请参考`Adaptive Communication Strategies to Achieve the Best Error-Runtime Trade-off in Local-Update SGD <https://arxiv.org/pdf/1810.08313.pdf>`_

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.adaptive_localsgd = True # by default this is false

adaptive_localsgd_configs
'''''''''
设置 AdaptiveLocalSGD 优化器的参数。用户可以配置 init_k_steps 和 begin_step 参数。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.adaptive_localsgd = True
  strategy.adaptive_localsgd_configs = {"init_k_steps": 1,
                                        "begin_step": 30}

**init_k_steps(int):** 自适应 localsgd 的初始训练步长。训练后，自适应 localsgd 方法将自动调整步长。默认值 1。

**begin_step(int):** 指定从第几个 step 之后进行 Adaptive LocalSGD 算法，默认值 1。

amp
'''''''''

是否启用自动混合精度训练。默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.amp = True # by default this is false

amp_configs
'''''''''

设置自动混合精度训练配置。为避免梯度 inf 或 nan，amp 会根据梯度值自动调整 loss scale 值。目前可以通过字典设置以下配置。

**init_loss_scaling(float):** 初始 loss scaling 值。默认值 32768。

**use_dynamic_loss_scaling(bool):** 是否动态调整 loss scale 值。默认 True。

**incr_every_n_steps(int):** 每经过 n 个连续的正常梯度值才会增大 loss scaling 值。默认值 1000。

**decr_every_n_nan_or_inf(int):** 每经过 n 个连续的无效梯度值(nan 或者 inf)才会减小 loss scaling 值。默认值 2。

**incr_ratio(float):** 每次增大 loss scaling 值的扩增倍数，其为大于 1 的浮点数。默认值 2.0。

**decr_ratio(float):** 每次减小 loss scaling 值的比例系数，其为小于 1 的浮点数。默认值 0.5。

**custom_white_list(list[str]):** 用户自定义 OP 开启 fp16 执行的白名单。

**custom_black_list(list[str]):** 用户自定义 OP 禁止 fp16 执行的黑名单。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.amp = True
  strategy.amp_configs = {
      "init_loss_scaling": 32768,
      "custom_white_list": ['conv2d']}

dgc
'''''''''

是否启用深度梯度压缩训练。更多信息请参考[Deep Gradient Compression](https://arxiv.org/abs/1712.01887)。默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.dgc = True  # by default this is false

dgc_configs
'''''''''

设置 dgc 策略的配置。目前用户可配置 rampup_begin_step，rampup_step，sparsity 参数。

**rampup_begin_step(int):** 梯度压缩的起点步。默认值 0。

**rampup_step(int):** 使用稀疏预热的时间步长。默认值为 1。例如：如果稀疏度为[0.75,0.9375,0.984375,0.996,0.999]，\
并且 rampup_step 为 100，则在 0~19 步时使用 0.75，在 20~39 步时使用 0.9375，依此类推。当到达 sparsity 数组末尾时，此后将会使用 0.999。

**sparsity(list[float]):** 从梯度 Tensor 中获取 top 个重要元素，比率为（1-当前稀疏度）。默认值为[0.999]。\
例如：如果 sparsity 为[0.99, 0.999]，则将传输 top [1%, 0.1%]的重要元素。

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.dgc = True
  strategy.dgc_configs = {"rampup_begin_step": 1252}

fp16_allreduce
'''''''''

是否使用 fp16 梯度 allreduce 训练。默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.fp16_allreduce = True  # by default this is false


sharding
'''''''''

是否开启 sharding 策略。sharding 实现了[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
中 ZeRO-DP 类似的功能，其通过将模型的参数和优化器状态在 ranks 间分片来支持更大模型的训练。

目前在混合并行(Hybrid parallelism) 模式下，sharding config 作为混合并行设置的统一入口来设置混合并行相关参数。

默认值：False

**代码示例**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.sharding = True

sharding_configs
'''''''''

设置 sharding 策略的参数。

**sharding_segment_strategy(float, optional):** 选择 sharding 中用来将前向反向 program 切 segments 的策略。目前可选策略有："segment_broadcast_MB" 和 "segment_anchors"。 segment 是 sharding 中引入的一个内部概念，目的是用来让通信和计算相互重叠掩盖（overlap）。默认值是 segment_broadcast_MB。

**segment_broadcast_MB(float, optional):** 根据 sharding 广播通信中的参数量来切 segments，仅当 sharding_segment_strategy = segment_broadcast_MB 时生效。sharding 会在前向和反向中引入参数广播，在该 segment 策略下，每当参数广播量达到 “segment_broadcast_MB”时，在 program 中切出一个 segment。该参数是一个经验值，最优值会受模型大小和网咯拓扑的影响。默认值是 32。

**segment_anchors(list):** 根据用户选定的锚点切割 segments，仅当 sharding_segment_strategy = segment_anchors 生效。该策略可以让用户更精确的控制 program 的切分，目前还在实验阶段。

**sharding_degree(int, optional):** sharding 并行数。sharding_degree=1 时，sharding 策略会被关闭。默认值是 8。

**gradient_merge_acc_step(int, optional):** 梯度累积中的累积步数。gradient_merge_acc_step=1 梯度累积会被关闭。默认值是 1。

**optimize_offload(bool, optional):** 优化器状态卸载开关。开启后会将优化器中的状态(moment) 卸载到 Host 的内存中，以到达节省 GPU 显存、支持更大模型的目的。开启后，优化器状态会在训练的更新阶段经历：预取-计算-卸载（offload）三个阶段，更新阶段耗时会增加。这个策略需要权衡显存节省量和训练速度，仅推荐在开启梯度累积并且累积步数较大时开启。因为累积步数较大时，训练中更新阶段的比例将远小于前向&反向阶段，卸载引入的耗时将不明显。

**dp_degree(int, optional):** 数据并行的路数。当 dp_degree>=2 时，会在内层并行的基础上，再引入 dp_degree 路 数据并行。用户需要保证 global_world_size = mp_degree * sharding_degree * pp_degree * dp_degree。默认值是 1。

**mp_degree(int, optional):** [仅在混合并行中使用] megatron 并行数。mp_degree=1 时，mp 策略会被关闭。默认值是 1。

**pp_degree(int, optional):** [仅在混合并行中使用] pipeline 并行数。pp_degree=1 时，pipeline 策略会被关闭。默认值是 1。

**pp_allreduce_in_optimize(bool, optional):** [仅在混合并行中使用] 在开启 pipeline 并行后，将 allreduce 操作从反向阶段移动到更新阶段。根据不同的网络拓扑，该选项会影响训练速度，该策略目前还在实验阶段。默认值是 False。


.. code-block:: python

  # sharding-DP, 2 nodes with 8 gpus per node
  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.sharding = True
  strategy.sharding_configs = {
      "sharding_segment_strategy": "segment_broadcast_MB",
      "segment_broadcast_MB": 32,
      "sharding_degree": 8,
      "dp_degree": 2,
      "gradient_merge_acc_step": 4,
      }
