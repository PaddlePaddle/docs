.. _cn_api_paddle_distributed_local_map:

local_map
-------------------------------

.. py:function:: paddle.distributed.local_map(func, out_placements, in_placements=None, process_mesh=None, reshard_inputs=False)

local_map 是一个函数装饰器，允许用户将分布式张量（DTensor）传递给为普通张量（Tensor）编写的函数。它通过提取分布式张量的本地分量，调用目标函数，并根据 out_placements 将输出包装为分布式张量来实现这一功能，通过自动处理张量转换，使得用户可以像编写单卡代码一样实现这些局部操作。


参数
:::::::::

    - **func** (Callable) - 要应用于分布式张量本地分片的函数
    - **out_placements** (list[list[dist.Placement]]) - 指定输出张量的分布策略。外层列表长度必须与函数输出数量匹配，每个内层列表描述对应输出张量的分布方式。对于非张量输出必须设为 None
    - **in_placements** (list[list[dist.Placement]] | None) - 指定输入张量的要求分布。如果指定，每个内层列表描述对应输入张量的分布要求。外层列表长度必须与输入张量数量匹配。对于不具有分布式属性的输入应设为 None,默认为 None
    - **process_mesh** (Optional[ProcessMesh]) - 计算设备网格。所有分布式张量必须位于同一个 process_mesh 上。如未指定则从输入张量推断
    - **reshard_inputs** (bool) - 当输入分布式张量的分布方式与要求的 in_placements 不匹配时,是否自动重分布。默认 False

返回
:::::::::

    返回一个可调用对象（Callable），该对象将 func 应用于输入分布式张量的每个本地分片，并根据返回值构造新的分布式张量。


异常抛出情况
:::::::::

    - **AssertionError** - 当输出分布策略的数量与函数输出的数量不匹配时抛出。
    - **AssertionError** - 当非张量输出指定了非 None 的分布策略时抛出。
    - **AssertionError** - 当 process_mesh 为 None 且没有分布式张量输入，但 out_placements 包含非 None 值时抛出。
    - **ValueError** - 当输入分布式张量的分布方式与要求的 in_placements 不匹配，且 reshard_inputs 为 False 时抛出。


代码示例
:::::::::

.. code-block:: python

    import paddle
    import paddle.distributed as dist
    from paddle import Tensor
    from paddle.distributed import ProcessMesh

    def custom_function(x):
        mask = paddle.zeros_like(x)
        if dist.get_rank() == 0:
            mask[1:3] = 1
        else:
            mask[4:7] = 1
        x = x * mask
        mask_sum = paddle.sum(x)
        mask_sum = mask_sum / mask.sum()
        return mask_sum

    dist.init_parallel_env()
    mesh = ProcessMesh([0, 1], dim_names=["x"])

    local_input = paddle.arange(0, 10, dtype='float32')
    local_input = local_input + dist.get_rank()

    input_dist = dist.auto_parallel.api.dtensor_from_local(
        local_input,
        mesh,
        [dist.Shard(0)]
    )

    # 使用 local_map 包装函数
    wrapped_func = dist.local_map(
        custom_function,
        out_placements=[[dist.Partial(dist.ReduceType.kRedSum)]],
        in_placements=[[dist.Shard(0)]],
        process_mesh=mesh
    )

    # 应用函数到分布式张量
    output_dist = wrapped_func(input_dist)

    # 收集并打印结果
    local_value = output_dist._local_value()
    gathered_values: list[Tensor] = []
    dist.all_gather(gathered_values, local_value)
    print(f"[Rank 0] local_value={gathered_values[0].item()}")
    # [Rank 0] local_value=1.5
    print(f"[Rank 1] local_value={gathered_values[1].item()}")
    # [Rank 1] local_value=6.0
    print(f"global_value (distributed)={output_dist.item()}")
    # global_value (distributed)=7.5
