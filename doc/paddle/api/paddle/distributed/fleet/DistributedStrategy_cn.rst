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
  **checkpoints(int)** Recompute策略的检查点，默认为空列表，也即不启用Recompute。

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
  **k_steps(int)** 参数更新的周期，默认为1

  **avg(bool)** 梯度的融合方式，有两种选择：
    - **sum**: 梯度求和
    - **avg**: 梯度求平均

纯测试 

测试1

设置 **distribute_strategy** 策略的配置。
  **k_steps(int)** 参数更新的周期，默认为1


测试2

设置 **distribute_strategy** 策略的配置。
  **k_steps(int)** 参数更新的周期，默认为1

  **avg(bool)** 梯度的融合方式，有两种选择：


测试3：

设置 distribute_strategy 策略的配置。
  **k_steps(int)** 参数更新的周期，默认为1

  **avg(bool)** 梯度的融合方式，有两种选择：
    - **sum**: 梯度求和
    - **avg**: 梯度求平均


测试4：

设置 **distribute_strategy** 策略的配置。
  k_steps(int) 参数更新的周期，默认为1

  avg(bool) 梯度的融合方式，有两种选择：
    - **sum**: 梯度求和
    - **avg**: 梯度求平均

测试5 

设置 distribute_strategy 策略的配置。
  k_steps(int) 参数更新的周期，默认为1

  avg(bool) 梯度的融合方式，有两种选择：
    - sum: 梯度求和
    - avg: 梯度求平均

测试6

设置 distribute_strategy 策略的配置。

  k_steps(int) 参数更新的周期，默认为1

  avg(bool) 梯度的融合方式，有两种选择：

    - sum: 梯度求和
    - avg: 梯度求平均

测试7 


设置 distribute_strategy 策略的配置。

  k_steps(int) 参数更新的周期，默认为1

测试8 

设置 distribute_strategy 策略的配置。
  k_steps(int) 参数更新的周期，默认为1