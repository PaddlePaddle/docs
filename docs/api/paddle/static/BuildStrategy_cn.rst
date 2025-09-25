.. _cn_api_paddle_static_BuildStrategy:

BuildStrategy
-------------------------------

.. py:class:: paddle.static.BuildStrategy

 ``BuildStrategy``  使用户更方便地控制  ``ParallelExecutor``  中计算图的建造方法，可通过设置  ``ParallelExecutor``  中的  ``BuildStrategy``  成员来实现此功能。

返回
:::::::::
BuildStrategy，一个 BuildStrategy 的实例。

代码示例
:::::::::

COPY-FROM: paddle.static.BuildStrategy

属性
::::::::::::
build_cinn_pass
'''''''''

str 类型。表示是否将计算图中的一些算子降级（lowering）为 CINN 算子来执行，这可以加速执行过程。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.build_cinn_pass

debug_graphviz_path
'''''''''

str 类型。表示以 graphviz 格式向文件中写入计算图的路径，有利于调试。默认值为空字符串。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.debug_graphviz_path


enable_auto_fusion
'''''''''

bool 类型。是否将子图（subgraph）融合成一个融合组（fusion_group）。目前我们仅支持融合由逐元素类（elementwise-like）算子组成的子图，例如无广播（broadcast）机制的 elementwise_add/mul 以及激活函数（activations）。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.enable_auto_fusion

fuse_adamw
'''''''''

bool 类型。表示是否将所有的 adamw 优化器与 multi_tensor_adam 进行融合，该操作可能会提升执行速度。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_adamw

fuse_bn_act_ops
'''''''''

bool 类型。表示是否融合批量归一化（batch_norm）和激活算子（activation_op），该操作可提升执行速度。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_bn_act_ops

fuse_bn_add_act_ops
'''''''''

bool 类型。表示是否融合批量归一化（batch_norm）、逐元素加法（elementwise_add）和激活算子（activation_op），该操作可提升执行速度。默认值为 True。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_bn_add_act_ops

fuse_broadcast_ops
'''''''''

bool 类型。表明是否融合(fuse) broadcast ops。该选项指在 Reduce 模式下有效，使程序运行更快。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_broadcast_ops

fuse_dot_product_attention
'''''''''

bool 类型。表示是否融合点积注意力（dot product attention），该操作可以提升执行速度。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_dot_product_attention

fuse_elewise_add_act_ops
'''''''''

bool 类型。表明是否融合(fuse) elementwise_add_op 和 activation_op。这会使整体执行过程更快。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_elewise_add_act_ops

fuse_gemm_epilogue
'''''''''

bool 类型。表示是否融合矩阵乘法算子（matmul_op）、逐元素加法算子（elementwise_add_op）和激活算子（activation_op），该操作可提升执行速度。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_gemm_epilogue

fuse_relu_depthwise_conv
'''''''''

bool 类型。表明是否融合(fuse) relu 和 depthwise_conv2d，节省 GPU 内存并可能加速执行过程。此选项仅适用于 GPU 设备。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_relu_depthwise_conv

fuse_resunit
'''''''''

bool 类型。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fuse_resunit

fused_attention
'''''''''

bool 类型。表示是否将整个多头注意力（multi-head attention）部分融合成一个算子（op），该操作可提升执行速度。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fused_attention

fused_feedforward
'''''''''

bool 类型。表示是否将整个前馈网络（feed_forward）部分融合成一个算子（op），该操作可提升执行速度。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.fused_feedforward

memory_optimize
'''''''''

bool 类型或 None。设为 True 时可用于减少总内存消耗，False 表示不使用，None 表示框架会自动选择使用或者不使用优化策略。当前，None 意味着当 GC 不能使用时，优化策略将被使用。默认为 None。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.memory_optimize

reduce_strategy
'''''''''

``static.BuildStrategy.ReduceStrategy``  类型。在  ``ParallelExecutor``  中，存在两种参数梯度聚合策略，即  ``AllReduce``  和  ``Reduce`` 。如果用户需要在所有执行设备上独立地进行参数更新，可以使用  ``AllReduce`` 。如果使用  ``Reduce``  策略，所有参数的优化将均匀地分配给不同的执行设备，随之将优化后的参数广播给其他执行设备。默认值为  ``AllReduce``  。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.reduce_strategy

sequential_run
'''''''''

bool 类型。该参数用于控制 StandaloneExecutor 是否按照 ProgramDesc 中定义的顺序来执行算子（ops）。默认值为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.sequential_run

sync_batch_norm
'''''''''

bool 类型。表示是否使用同步的批正则化，即在训练阶段通过多个设备同步均值和方差。当前的实现不支持 FP16 训练和 CPU。并且目前 **仅支持** 仅在一台机器上进行同步式批正则。默认为 False。

**代码示例**

COPY-FROM: paddle.static.BuildStrategy.sync_batch_norm
