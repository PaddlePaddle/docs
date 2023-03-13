.. _api_fluid_dygraph_prepare_context:

prepare_context
---------------

.. py:class:: paddle.fluid.dygraph.prepare_context(strategy=None)




该 API 是进行多进程多卡训练的环境配置接口，接受一个 ParallelStrategy 结构体变量作为输入。当 strategy 属性中的 nums_trainer 小于 2 时，API 会直接返回，当 nums_trainer 大于 1 且为 CUDAPlace 时，由于目前动态图模式仅支持 GPU 多卡训练，仅能配置 NCCL 多卡训练的环境，所以此时会对 NCCL 环境进行配置，具体内容包括：生成 NCCL ID，并广播至参与训练的各进程，用于支持的处理器同步操作，创建并配置 NCCL 通信器等。

参数
::::::::::::

  - **strategy** (ParallelStrategy，可选) – 该参数是配置储存多进程多卡训练配置信息的结构体变量，其具体成员包括：trainer 节点的个数，当前 trainer 节点的 ID，所有 trainer 节点的 endpoint，当前节点的 endpoint。当输入为 None 时，会调用 PallelStrategy 构造函数初始化 strategy，此时，strategy 的属性值为 PallelStrategy 结构体的默认值，接着 strategy 的属性会被环境变量中的对应值覆盖。默认值为 None。

返回
::::::::::::
一个属性配置后的 ParallelStrategy 结构体变量。

返回类型
::::::::::::
实例（ParallelStrategy）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.dygraph.prepare_context
