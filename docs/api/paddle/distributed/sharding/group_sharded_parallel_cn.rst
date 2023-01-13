.. _cn_api_distributed_sharding_group_sharded_parallel:

group_sharded_parallel
-------------------------------


.. py:function:: paddle.distributed.sharding.group_sharded_parallel(model, optimizer, level, scaler=None, group=None, offload=False, sync_buffers=False, buffer_max_size=2**23, segment_size=2**20, sync_comm=False, dp_group=None, exclude_layer=None)

使用 group_sharded_parallel 可以对模型、优化器和 GradScaler 做 group sharded 配置。level 有三个字符串选项，分别是'os','os_g','p_g_os'，分别对应优化器状态切分、优化器状态+梯度切分、参数+梯度+优化器状态切分三种不同的使用场景。
通常情况下优化器状态+梯度切分实际上是优化器状态切分的一种再优化，所以实现上可以用优化器状态+梯度切分实现优化器状态切分。


参数
:::::::::
    - **model** (Layer) - 需要使用 group sharded 的模型。
    - **optimizer** (Optimizer) - 需要使用 group sharded 的优化器。
    - **level** (str) - 选择 group sharded 的级别，分别有'os','os_g','p_g_os'。
    - **scaler** (GradScaler，可选) - 如果使用 AMP 混合精度，需要传入 GradScaler，默认为 None，表示不使用 GradScaler。
    - **group** (Group，可选) - 工作的进程组编号，默认为 None，表示采用默认环境 Group。
    - **offload** (bool，可选) - 是否使用 offload 缓存功能，默认为 False，表示不使用 offload 功能。
    - **sync_buffers** (bool，可选) - 是否需要同步模型 buffers，一般在有注册模型 buffers 时才使用，默认为 False，表示不同步模型 buffers。
    - **buffer_max_size** (int，可选) - 在'os_g'模式中会对梯度进行聚合，此选项指定聚合 buffer 的大小，指定越大则占用显存也越多，默认为 2**23，表示聚合 buffer 的维度为 2**23。
    - **segment_size** (int，可选) - 在'p_g_os'模式中会对参数进行切分，此选项指定最小切分参数大小，默认为 2**20，表示最小被切分参数的维度为 2**20。
    - **sync_comm** (bool，可选) - 在'p_g_os'模式中是否采用同步通信，默认为 False，表示使用异步通信流。
    - **dp_group** (Group，可选) - 数据并行（data parallel）通信组，支持 sharding 的 stage2 和 stage3 和数据并行一起混合使用。
    - **exclude_layer** (list，可选) - 在 sharding stage3 中，可以设置某些 layer 的参数不切分（通过 layer 的类型或 id 控制），例如：exclude_layer=["GroupNorm", id(model.gpt.linear)]。

返回
:::::::::
group sharded 配置后的 model，optimizer 和 scaler

代码示例
:::::::::
COPY-FROM: paddle.distributed.sharding.group_sharded_parallel
