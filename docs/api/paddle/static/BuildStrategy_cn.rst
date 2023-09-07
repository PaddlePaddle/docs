.. _cn_api_paddle_static_BuildStrategy:

BuildStrategy
-------------------------------

.. py:class:: paddle.static.BuildStrategy

``BuildStrategy`` 使用户更方便地控制 :ref:`cn_api_fluid_ParallelExecutor` 中计算图的建造方法，可通过设置 ``ParallelExecutor`` 中的 ``BuildStrategy`` 成员来实现此功能。

返回
:::::::::
BuildStrategy，一个 BuildStrategy 的实例。

代码示例
:::::::::

COPY-FROM: paddle.static.BuildStrategy

属性
::::::::::::
debug_graphviz_path
'''''''''

str 类型。表示以 graphviz 格式向文件中写入计算图的路径，有利于调试。默认值为空字符串。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.debug_graphviz_path


enable_sequential_execution
'''''''''

bool 类型。如果设置为 True，则算子的执行顺序将与算子定义的执行顺序相同。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.enable_sequential_execution

fuse_broadcast_ops
'''''''''

bool 类型。表明是否融合(fuse) broadcast ops。该选项指在 Reduce 模式下有效，使程序运行更快。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_broadcast_ops

fuse_elewise_add_act_ops
'''''''''

bool 类型。表明是否融合(fuse) elementwise_add_op 和 activation_op。这会使整体执行过程更快。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_elewise_add_act_ops

fuse_relu_depthwise_conv
'''''''''

bool 类型。表明是否融合(fuse) relu 和 depthwise_conv2d，节省 GPU 内存并可能加速执行过程。此选项仅适用于 GPU 设备。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_relu_depthwise_conv

gradient_scale_strategy
'''''''''

``paddle.static.BuildStrategy.GradientScaleStrategy`` 类型。在 ``ParallelExecutor`` 中，存在三种定义 loss 对应梯度( *loss@grad* )的方式，分别为 ``CoeffNumDevice``, ``One`` 与 ``Customized``。默认情况下，``ParallelExecutor`` 根据设备数目来设置 *loss@grad*。如果用户需要自定义 *loss@grad*，可以选择 ``Customized`` 方法。默认为 ``CoeffNumDevice`` 。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.gradient_scale_strategy

memory_optimize
'''''''''

bool 类型或 None。设为 True 时可用于减少总内存消耗，False 表示不使用，None 表示框架会自动选择使用或者不使用优化策略。当前，None 意味着当 GC 不能使用时，优化策略将被使用。默认为 None。

reduce_strategy
'''''''''

``static.BuildStrategy.ReduceStrategy`` 类型。在 ``ParallelExecutor`` 中，存在两种参数梯度聚合策略，即 ``AllReduce`` 和 ``Reduce``。如果用户需要在所有执行设备上独立地进行参数更新，可以使用 ``AllReduce``。如果使用 ``Reduce`` 策略，所有参数的优化将均匀地分配给不同的执行设备，随之将优化后的参数广播给其他执行设备。
默认值为 ``AllReduce`` 。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.reduce_strategy

remove_unnecessary_lock
'''''''''

bool 类型。设置 True 会去除 GPU 操作中的一些锁操作，``ParallelExecutor`` 将运行得更快，默认为 True。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.remove_unnecessary_lock

sync_batch_norm
'''''''''

bool 类型。表示是否使用同步的批正则化，即在训练阶段通过多个设备同步均值和方差。当前的实现不支持 FP16 训练和 CPU。并且目前**仅支持**仅在一台机器上进行同步式批正则。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.sync_batch_norm
