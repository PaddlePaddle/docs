.. _cn_api_fluid_BuildStrategy:

BuildStrategy
-------------------------------

.. py:class::  paddle.fluid.BuildStrategy

``BuildStrategy`` 使用户更精准地控制 ``ParallelExecutor`` 中SSA图的建造方法。可通过设置 ``ParallelExecutor`` 中的 ``BuildStrategy`` 成员来实现此功能。

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce


.. py:attribute:: debug_graphviz_path

str类型。它表明了以graphviz格式向文件中写入SSA图的路径，有利于调试。 默认值为""。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.debug_graphviz_path = ""


.. py:attribute:: enable_sequential_execution

类型是BOOL。 如果设置为True，则ops的执行顺序将与program中的执行顺序相同。 默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_sequential_execution = True


.. py:attribute:: fuse_broadcast_ops
     
bool类型。它表明了是否融合（fuse）broadcast ops。值得注意的是，在Reduce模式中，融合broadcast ops可以使程序运行更快，因为这个过程等同于延迟执行所有的broadcast ops。在这种情况下，所有的nccl streams仅用于一段时间内的NCCLReduce操作。默认为False。
     
.. py:attribute:: fuse_elewise_add_act_ops

bool类型。它表明了是否融合（fuse）elementwise_add_op和activation_op。这会使整体执行过程更快一些。默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True


.. py:attribute:: fuse_relu_depthwise_conv

BOOL类型，fuse_relu_depthwise_conv指示是否融合relu和depthwise_conv2d，它会节省GPU内存并可能加速执行过程。 此选项仅适用于GPU设备。 默认为False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_relu_depthwise_conv = True

.. py:attribute:: gradient_scale_strategy

str类型。在 ``ParallelExecutor`` 中，存在三种定义 *loss@grad* 的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下， ``ParallelExecutor`` 根据设备数目来设置 *loss@grad* 。如果你想自定义 *loss@grad* ，你可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.gradient_scale_strategy = True

.. py:attribute:: memory_optimize

bool类型。设为True时可用于减少总内存消耗。为实验性属性，一些变量可能会被优化策略重用/移除。如果你需要在使用该特征时获取某些变量，请把变量的persistable property设为True。默认为False。

.. py:attribute:: reduce_strategy

str类型。在 ``ParallelExecutor`` 中，存在两种减少策略（reduce strategy），即 ``AllReduce`` 和 ``Reduce`` 。如果你需要在所有执行场所上独立地进行参数优化，可以使用 ``AllReduce`` 。反之，如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行场所，随之将优化后的参数广播给其他执行场所。在一些模型中， ``Reduce`` 策略执行速度更快一些。默认值为 ``AllReduce`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

.. py:attribute:: remove_unnecessary_lock

BOOL类型。如果设置为True, GPU操作中的一些锁将被释放，ParallelExecutor将运行得更快，默认为 True。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.remove_unnecessary_lock = True


.. py:attribute:: sync_batch_norm

类型为bool，sync_batch_norm表示是否使用同步的批正则化，即在训练阶段通过多个设备同步均值和方差。

当前的实现不支持FP16培训和CPU。仅在一台机器上进行同步式批正则，不适用于多台机器。

默认为 False。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.sync_batch_norm = True


