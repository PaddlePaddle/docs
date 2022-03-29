.. _cn_api_fluid_IpuStrategy:

IpuStrategy
-------------------------------


.. py:class:: paddle.static.IpuStrategy()


``IpuStrategy`` 使用户能更精准地控制 :ref:`cn_api_fluid_IpuCompiledProgram` 中计算图的建造方法。


返回
:::::::::
    IpuStrategy实例

代码示例
::::::::::

COPY-FROM: paddle.static.IpuStrategy

方法
::::::::::::
set_graph_config(self, num_ipus, is_training, micro_batch_size, enable_manual_shard)
'''''''''

该接口用于向IpuStrategy实例传递IPU构图的Graph配置。

**参数**
    - **num_ipus** (int，可选)- 指定IPU devices的个数，默认值为1，表示仅用一个IPU。
    - **is_training** (bool，可选)- 声明是训练还是推理，默认值为True，表示使用训练模式。
    - **micro_batch_size** (int，可选)- 当计算图输入的micro_batch_size可变时，指定计算图中输入micro_batch_size，默认值为1，表示如果micro_batch_size可变，将默认置1。
    - **enable_manual_shard** (bool，可选)- 是否使能分割计算图到不同IPU进行运算。仅支持当num_ipus > 1时，enable_manual_shard可以置为True。默认值为False，表示不使能该功能。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_graph_config

set_pipelining_config(self, enable_pipelining, batches_per_step, enable_gradient_accumulation, accumulation_factor)
'''''''''

该接口用于向IpuStrategy实例传递IPU构图的子图数据流水线配置。

**参数**
    - **enable_pipelining** (bool，可选)- 是否使能子图之间的数据流水线。仅支持当enable_manual_shard=True时，enable_pipelining可以置为True。默认值为False，表示不使能该功能。
    - **batches_per_step** (int，可选)- 指定数据流水线每次运算多少个batch的数据。默认值为1，表示不使能数据流水线功能。
    - **enable_gradient_accumulation** (bool，可选)- 是否使能梯度累积，只用于训练模式。默认值为Flase，表示不使能梯度累积功能。
    - **accumulation_factor** (int，可选)- 指定累积运算多少个batch更新一次权重。默认值为1，表示不使能权重累积更新功能。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_pipelining_config

set_precision_config(self, enable_fp16)
'''''''''

该接口用于向IpuStrategy实例传递IPU构图的精度配置。

**参数**
    - **enable_fp16** (bool)- 是否使能fp16运算模式并将fp32转换为fp16。默认值为False，表示不使能fp16运算模式。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_precision_config

add_custom_op(self, paddle_op, popart_op, domain, version)
'''''''''

该接口用于向IpuStrategy实例传递PopART自定义算子的信息。

**参数**
    - **paddle_op** (str)- 待添加的Paddle自定义算子在的名称，根据Paddle自定义算子的定义设置此参数。
    - **popart_op** (str，可选)- 待添加的PopART自定义算子的名称，默认值为None，表示和paddle_op相同，根据PopART自定算子的定义设置此参数。
    - **domain** (str，可选)- 待添加的PopART自定义算子的domain属性，默认值为"custom.ops"，根据PopART自定算子的定义设置此参数。
    - **version** (int，可选)- 待添加的PopART自定义算子的version属性，默认值为1，根据PopART自定算子的定义设置此参数。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.add_custom_op

set_options(self, options)
'''''''''

批量向IpuStrategy实例传递参数。

**参数**
    - **options** (dict)- 需要传递的参数字典。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.set_options

get_option(self, option)
'''''''''

获取IpuStrategy实例的某一参数。

**参数**
    - **option** (str)- 需要获取参数的名称。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.get_option

enable_pattern(self, pattern)
'''''''''

启用某一PopART Pattern。

**参数**
    - **pattern** (str)- 需要开启的Pattern名称。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.enable_pattern

disable_pattern(self, pattern)
'''''''''

禁用某一PopART Pattern。

**参数**
    - **pattern** (str)- 需要禁用的Pattern名称。

**代码示例**

COPY-FROM: paddle.static.IpuStrategy.disable_pattern

属性
::::::::::::
属性
::::::::::::
num_ipus
'''''''''

返回IpuStrategy实例中的IPU设备个数，类型为 ``Int``。

is_training
'''''''''

返回IpuStrategy实例中的计算模式是训练模式或推理模式，类型为 ``Bool``。

enable_pipelining
'''''''''

返回IpuStrategy实例中是否使能数据流水线功能，类型为 ``Bool``。

enable_fp16
'''''''''

返回IpuStrategy实例中是否使能float16计算图，类型为 ``Bool``。
