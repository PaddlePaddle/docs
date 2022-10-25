.. _cn_api_fluid_IpuStrategy:

IpuStrategy
-------------------------------


.. py:class:: paddle.static.IpuStrategy()


``IpuStrategy`` 使用户能更精准地控制 :ref:`cn_api_fluid_IpuCompiledProgram` 中计算图的建造方法。


返回
:::::::::
IpuStrategy 实例。

代码示例
::::::::::

COPY-FROM: paddle.static.IpuStrategy

方法
::::::::::::
set_graph_config(self, num_ipus, is_training, micro_batch_size, enable_manual_shard)
'''''''''

向 IpuStrategy 实例传递 IPU 构图的 Graph 配置。

**参数**

    - **num_ipus** (int，可选)- 指定 IPU devices 的个数，默认值为 1，表示仅用一个 IPU。
    - **is_training** (bool，可选)- 声明是训练还是推理，默认值为 True，表示使用训练模式。
    - **micro_batch_size** (int，可选)- 当计算图输入的 micro_batch_size 可变时，指定计算图中输入 micro_batch_size，默认值为 1，表示如果 micro_batch_size 可变，将默认置 1。
    - **enable_manual_shard** (bool，可选)- 是否使能分割计算图到不同 IPU 进行运算。仅支持当 num_ipus > 1 时，enable_manual_shard 可以置为 True。默认值为 False，表示不使能该功能。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_graph_config

set_pipelining_config(self, enable_pipelining, batches_per_step, enable_gradient_accumulation, accumulation_factor)
'''''''''

向 IpuStrategy 实例传递 IPU 构图的子图数据流水线配置。

**参数**

    - **enable_pipelining** (bool，可选)- 是否使能子图之间的数据流水线。仅支持当 enable_manual_shard=True 时，enable_pipelining 可以置为 True。默认值为 False，表示不使能该功能。
    - **batches_per_step** (int，可选)- 指定数据流水线每次运算多少个 batch 的数据。默认值为 1，表示不使能数据流水线功能。
    - **enable_gradient_accumulation** (bool，可选)- 是否使能梯度累积，只用于训练模式。默认值为 Flase，表示不使能梯度累积功能。
    - **accumulation_factor** (int，可选)- 指定累积运算多少个 batch 更新一次权重。默认值为 1，表示不使能权重累积更新功能。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_pipelining_config

set_precision_config(self, enable_fp16)
'''''''''

向 IpuStrategy 实例传递 IPU 构图的精度配置。

**参数**

    - **enable_fp16** (bool)- 是否使能 fp16 运算模式并将 fp32 转换为 fp16。默认值为 False，表示不使能 fp16 运算模式。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_precision_config

add_custom_op(self, paddle_op, popart_op, domain, version)
'''''''''

向 IpuStrategy 实例传递 PopART 自定义算子的信息。

**参数**

    - **paddle_op** (str)- 待添加的 Paddle 自定义算子在的名称，根据 Paddle 自定义算子的定义设置此参数。
    - **popart_op** (str，可选)- 待添加的 PopART 自定义算子的名称，默认值为 None，表示和 paddle_op 相同，根据 PopART 自定算子的定义设置此参数。
    - **domain** (str，可选)- 待添加的 PopART 自定义算子的 domain 属性，默认值为"custom.ops"，根据 PopART 自定算子的定义设置此参数。
    - **version** (int，可选)- 待添加的 PopART 自定义算子的 version 属性，默认值为 1，根据 PopART 自定算子的定义设置此参数。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.add_custom_op

set_options(self, options)
'''''''''

批量向 IpuStrategy 实例传递参数。

**参数**

    - **options** (dict)- 需要传递的参数字典。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_options

get_option(self, option)
'''''''''

获取 IpuStrategy 实例的某一参数。

**参数**

    - **option** (str)- 需要获取参数的名称。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.get_option

enable_pattern(self, pattern)
'''''''''

启用某一 PopART Pattern。

**参数**

    - **pattern** (str)- 需要开启的 Pattern 名称。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.enable_pattern

disable_pattern(self, pattern)
'''''''''

禁用某一 PopART Pattern。

**参数**

    - **pattern** (str)- 需要禁用的 Pattern 名称。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.disable_pattern

register_patch(self)
'''''''''

注册 patch function 以支持 IPU 上的动转静功能。该函数仅应在 IPU 动转静时使用，注册的函数会影响原动转静的逻辑，可通过``release_patch``释放注册的函数。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.register_patch

release_patch(self)
'''''''''

释放 IPU 动转静所注册的函数。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.release_patch

set_optimizer(self, optimizer)
'''''''''

在 IPU 动转静时向 IpuStrategy 实例设置 optimizer。

**参数**

    - **optimizer** (Optimizer)- 需要设置的 Optimizer 实例。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_optimizer

parse_optimizer(self, optimizer)
'''''''''

解析 IPU 动转静所需要的优化器参数，接收优化器实例并返回动转静所需要的优化器属性，当前仅支持解析学习率。

**参数**

    - **optimizer** (Optimizer)- 需要解析的 Optimizer 实例。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.parse_optimizer

属性
::::::::::::
num_ipus
'''''''''

返回 IpuStrategy 实例中的 IPU 设备个数，类型为 ``Int``。

is_training
'''''''''

返回 IpuStrategy 实例中的计算模式是训练模式或推理模式，类型为 ``Bool``。

enable_pipelining
'''''''''

返回 IpuStrategy 实例中是否使能数据流水线功能，类型为 ``Bool``。

enable_fp16
'''''''''

返回 IpuStrategy 实例中是否使能 float16 计算图，类型为 ``Bool``。
