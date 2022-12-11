.. _api_fluid_dygraph_prepare_context:

prepare_context
---------------

.. py:class:: paddle.fluid.dygraph.prepare_context(strategy=None)




该API是进行多进程多卡训练的环境配置接口，接受一个ParallelStrategy结构体变量作为输入。当strategy属性中的nums_trainer小于2时，API会直接返回，当nums_trainer大于1且为CUDAPlace时，由于目前动态图模式仅支持GPU多卡训练，仅能配置NCCL多卡训练的环境，所以此时会对NCCL环境进行配置，具体内容包括：生成NCCL ID，并广播至参与训练的各进程，用于支持的处理器同步操作，创建并配置NCCL通信器等。

参数
::::::::::::

  - **strategy** (ParallelStrategy，可选) – 该参数是配置储存多进程多卡训练配置信息的结构体变量，其具体成员包括：trainer节点的个数，当前trainer节点的ID，所有trainer节点的endpoint，当前节点的endpoint。当输入为None时，会调用PallelStrategy构造函数初始化strategy，此时，strategy的属性值为PallelStrategy结构体的默认值，接着strategy的属性会被环境变量中的对应值覆盖。默认值为None。

返回
::::::::::::
一个属性配置后的ParallelStrategy结构体变量。

返回类型
::::::::::::
实例（ParallelStrategy）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.dygraph.prepare_context