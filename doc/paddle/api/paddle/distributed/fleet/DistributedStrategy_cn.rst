.. _cn_api_distributed_fleet_DistributedStrategy:

DistributedStrategy
-------------------------------

.. py:class:: paddle.distributed.fleet.DistributedStrategy


属性
::::::::::::

.. py:attribute:: recompute

是否启用Recompute来优化内存空间，默认值：False

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.recompute = True
  # suppose x and y are names of checkpoint tensors for recomputation
  strategy.recompute_configs = {"checkpoints": ["x", "y"]}


.. py:attribute:: recompute_configs

设置Recompute策略的配置。目前来讲，用户使用Recompute策略时，必须配置 checkpoints 参数。

**checkpoints(int):** Recompute策略的检查点，默认为空列表，也即不启用Recompute。

.. py:attribute:: pipeline

是否启用Pipeline并行。目前，主要实现单机多GPU间的Pipeline并行和多机间的数据并行。Pipeline信息由用户定义程序中的device_guard确定。

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.pipeline = True


.. py:attribute:: pipeline_configs

设置Pipeline策略的配置。Pipeline策略下，神经网络的不同层在不同的GPU设备。相邻的GPU设备间有用于同步隐层Tensor的队列。Pipeline并行包含多种生产者-消费者形式的硬件对，如GPU-CPU、CPU-GPU、GPU-XPU。加速PIpeline并行的最佳方式是减少Tensor队列中的Tensor大小，这样生产者可以更快的为下游消费者提供数据。

**micro_batch (int):** 每个用户定义的mini-batch中包含的更小的micro-batch的数量。

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.pipeline = True
  strategy.pipeline_configs = {"micro_batch": 12}


.. py:attribute:: gradient_merge

梯度累加，是一种大Batch训练的策略。添加这一策略后，模型的参数每过 **k_steps** 步更新一次，
**k_steps** 是用户定义的步数。在不更新参数的步数里，Paddle只进行前向、反向网络的计算；
在更新参数的步数里，Paddle执行优化网络，通过特定的优化器（比如SGD、Adam），
将累加的梯度应用到模型参数上。

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.gradient_merge = True
  strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}  

.. py:attribute:: gradient_merge_configs

设置 **distribute_strategy** 策略的配置。

**k_steps(int):** 参数更新的周期，默认为1

**avg(bool):** 梯度的融合方式，有两种选择：

- **sum**: 梯度求和
- **avg**: 梯度求平均

.. py:attribute:: lars
是否使用LARS optimizer，默认值：False

**示例代码**

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
.. py:attribute:: lars_configs
设置LARS优化器的参数。用户可以配置 lars_coeff，lars_weight_decay，epsilon，exclude_from_weight_decay 参数。

**lars_coeff(float):** lars 系数，[原论文](https://arxiv.org/abs/1708.03888) 中的 trust coefficient。 默认值是 0.001.

**lars_weight_decay(float):** lars 公式中 weight decay 系数。 默认值是 0.0005.

**exclude_from_weight_decay(list[str]):** 不应用 weight decay 的 layers 的名字列表，某一layer 的name 如果在列表中，这一layer 的 lars_weight_decay将被置为 0. 默认值是 None.

**epsilon(float):** 一个小的浮点值，目的是维持数值稳定性，避免 lars 公式中的分母为零。 默认值是 0.

.. py:attribute:: lamb
是否使用LAMB optimizer，默认值：False

**示例代码**

.. code-block:: python
  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.lamb = True
  strategy.lamb_configs = {
      'lamb_weight_decay': 0.01,
      'exclude_from_weight_decay': [],
  }
.. py:attribute:: lamb_configs
设置LAMB优化器的参数。用户可以配置 lamb_weight_decay，exclude_from_weight_decay 参数。

**lamb_weight_decay(float):** lars 公式中 weight decay 系数。 默认值是 0.01.

**exclude_from_weight_decay(list[str]):** 不应用 weight decay 的 layers 的名字列表，某一layer 的name 如果在列表中，这一layer 的 lamb_weight_decay将被置为 0. 默认值是 None.

.. py:attribute:: localsgd
是否使用LocalSGD optimizer，默认值：False。更多的细节请参考[Don't Use Large Mini-Batches, Use Local SGD](https://arxiv.org/pdf/1808.07217.pdf)

**示例代码**

.. code-block:: python  

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.localsgd = True # by default this is false


.. py:attribute:: localsgd_configs
设置LocalSGD优化器的参数。用户可以配置k_steps和begin_step参数。

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.localsgd = True
  strategy.localsgd_configs = {"k_steps": 4,
                                "begin_step": 30}

**k_steps(int):** 训练过程中的全局参数更新间隔，默认值1。

**begin_step(int):** 指定从第几个step之后进行local SGD算法，默认值1。

.. py:attribute:: adaptive_localsgd
是否使用AdaptiveLocalSGD optimizer，默认值：False。更多的细节请参考[Adaptive Communication Strategies to Achieve the Best Error-Runtime Trade-off in Local-Update SGD](https://arxiv.org/pdf/1810.08313.pdf)

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.adaptive_localsgd = True # by default this is false

.. py:attribute:: adaptive_localsgd_configs
设置AdaptiveLocalSGD优化器的参数。用户可以配置init_k_steps和begin_step参数。

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.adaptive_localsgd = True
  strategy.adaptive_localsgd_configs = {"init_k_steps": 1,
                                        "begin_step": 30}

**init_k_steps(int):** 自适应localsgd的初始训练步长。训练后，自适应localsgd方法将自动调整步长。 默认值1。

**begin_step(int):** 指定从第几个step之后进行Adaptive LocalSGD算法，默认值1。
