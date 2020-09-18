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

是否为分布式训练启用Pipeline并行。目前，主要实现单机多GPU间的流水线并行和多机间的数据并行。
流水线信息用户定义的程序中的device_guard确定。

**示例代码**

.. code-block:: python

  import paddle.distributed.fleet as fleet
  strategy = fleet.DistributedStrategy()
  strategy.pipeline = True


.. py:attribute:: pipeline_configs

设置Pipeline策略的配置。Pipeline策略下，神经网络的不同层在不同的GPU设备。相邻的GPU设备间有用于
同步隐层Tensor的队列。Pipeline并行包含多种生产者-消费者形式的硬件对，如GPU-CPU、CPU-GPU、GPU-XPU。
加速PIpeline并行的最佳方式是减少Tensor队列中的Tensor大小，这样生产者可以更快的为下游消费者提供数据。

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
