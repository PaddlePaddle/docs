.. _cn_api_paddle_distributed_local_map:

local_map
-------------------------------

.. py:function:: paddle.distributed.local_map(func, out_placements, in_placements=None, process_mesh=None, reshard_inputs=False)

local_map 是一个函数装饰器，用于在分布式训练中实现局部计算操作。它允许用户将分布式张量传递给为普通张量编写的函数，通过自动处理张量转换，使得用户可以像编写单卡代码一样实现这些局部操作。

相比于 LocalLayer，local_map 具有以下优势：

1. 使用更简便：直接包装普通函数，无需继承 Layer 类
2. 更灵活的输入：支持混合输入(分布式张量、普通张量、非张量参数)
3. 自动reshard：支持自动 reshard 输入张量
4. 智能推导：若process_mesh未指定，则从输入张量自动推导 process_mesh
5. 运行模式兼容：支持动态图和静态图
6. 通用性更强：可处理任意 Python 函数，不限于 Layer 的forward方法
7. 灵活的输出：支持混合输出(分布式张量、普通张量、非张量值)

参数
:::::::::

    - **func** (Callable) - 要应用于分布式张量本地分片的函数
    - **out_placements** (list[list[dist.Placement]]) - 指定输出张量的分布策略。每个元素是一个Placement列表,描述对应输出张量的分布方式。对于不具有分布式属性的输出应设为None
    - **in_placements** (list[list[dist.Placement]] | None) - 指定输入张量的要求分布。每个元素是一个Placement列表,描述对应输入张量的分布要求.对于不具有分布式属性的输入应设为None,默认为None
    - **process_mesh** (Optional[ProcessMesh]) - 计算设备网格。如未指定则从输入张量推断
    - **reshard_inputs** (bool) - 当输入张量分布不符合要求时是否自动重分布。默认 False

返回
:::::::::

包装后的函数，接受分布式张量输入并返回指定分布的输出。

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

使用场景
:::::::::

local_map 适用于但不限于以下场景:

1. 带 mask 的 loss 计算：需要在每张卡上独立计算 masked token 的 loss
2. MoE (混合专家模型)相关计算：
   - aux_loss 计算：基于每张卡上专家分配到的局部 token 数进行计算
   - z_loss 计算：对每张卡上的 logits 独立计算 z_loss
   - 张量 reshape 操作：在局部维度上进行 shape 变换
3. 需要对分布式张量应用普通张量函数的场景
4. 需要混合处理分布式张量和普通张量的场景

注意事项
:::::::::

1. 输出必须指定正确的分布策略以确保结果正确性
2. 在函数中可以像单卡编程一样使用常规的 tensor 操作
3. 计算结果会自动根据分布策略进行聚合，无需手动添加通信操作
4. 当指定 in_placements 时，输入张量的分布必须匹配要求，除非启用 reshard_inputs
5. 所有分布式张量必须在同一个 process_mesh 上
