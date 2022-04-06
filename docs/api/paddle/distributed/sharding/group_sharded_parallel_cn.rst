.. _cn_api_distributed_sharding_group_sharded_parallel:

group_sharded_parallel
-------------------------------


.. py:function:: paddle.distributed.sharding.group_sharded_parallel(model, optimizer, level, scaler=None, group=None, offload=False, sync_buffers=False, buffer_max_size=2**23, segment_size=2**20, sync_comm=False)

使用group_sharded_parallel可以对模型、优化器和GradScaler做group sharded配置。level有三个字符串选项，分别是'os','os_g','p_g_os',分别对应优化器状态切分、优化器状态+梯度切分、参数+梯度+优化器状态切分三种不同的使用场景。
通常情况下优化器状态+梯度切分实际上是优化器状态切分的一种再优化，所以实现上可以用优化器状态+梯度切分实现优化器状态切分。


参数
:::::::::
    - model (Layer) - 需要使用group sharded的模型。
    - optimizer (Optimizer) - 需要使用group sharded的优化器。
    - level (str) - 选择group sharded的级别，分别有'os','os_g','p_g_os'。
    - scaler (GradScaler，可选) - 如果使用AMP混合精度，需要传入GradScaler，默认为None，表示不使用GradScaler。
    - group (Group，可选) - 工作的进程组编号，默认为None，表示采用默认环境Group。
    - offload (bool，可选) - 是否使用offload缓存功能，默认为False，表示不使用offload功能。
    - sync_buffers (bool，可选) - 是否需要同步模型buffers，一般在有注册模型buffers时才使用，默认为False，表示不同步模型buffers。
    - buffer_max_size (int，可选) - 在'os_g'模式中会对梯度进行聚合，此选项指定聚合buffer的大小，指定越大则占用显存也越多，默认为2**23，表示聚合buffer的维度为2**23。
    - segment_size (int，可选) - 在'p_g_os'模式中会对参数进行切分，此选项指定最小切分参数大小，默认为2**20，表示最小被切分参数的维度为2**20。
    - sync_comm (bool，可选) - 在'p_g_os'模式中是否采用同步通信，默认为False，表示使用异步通信流。

返回
:::::::::
group sharded配置后的model，optimizer和scaler

代码示例
:::::::::
COPY-FROM: paddle.distributed.sharding.group_sharded_parallel
     