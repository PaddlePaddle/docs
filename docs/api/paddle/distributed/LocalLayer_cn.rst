.. _cn_api_paddle_distributed_LocalLayer:

LocalLayer
-------------------------------

.. py:class:: paddle.distributed.LocalLayer(out_dist_attrs)

LocalLayer 用于在分布式训练中实现局部计算操作。在自动并行训练中,某些操作(如带 mask 的 loss 计算、MoE 相关计算等)需要在每张卡上独立进行局部计算,而不是直接在全局分布式张量上计算。LocalLayer 通过自动处理张量转换,使得用户可以像编写单卡代码一样实现这些局部操作。

参数
:::::::::

    - **out_dist_attrs** (list[tuple[ProcessMesh, list[Placement]]]) - 指定输出张量的分布策略。每个元素是一个元组,包含:

      - ProcessMesh: 计算设备网格,定义计算资源的拓扑结构
      - list[Placement]: 张量分布方式的列表,描述如何将局部计算结果转换回分布式张量

**代码示例**

.. code-block:: python

    import paddle
    import paddle.distributed as dist
    from paddle.distributed import Placement, ProcessMesh, LocalLayer

    class CustomLayer(dist.LocalLayer):
        def __init__(self, out_dist_attrs):
            super().__init__(out_dist_attrs)
            self.local_result = paddle.to_tensor(0.0)
        def forward(self, x):
            mask = paddle.zeros_like(x)
            if dist.get_rank() == 0:
                mask[1:3] = 1
            else:
                mask[4:7] = 1
            x = x * mask
            mask_sum = paddle.sum(x)
            mask_sum = mask_sum / mask.sum()
            self.local_result = mask_sum
            return mask_sum

    dist.init_parallel_env()
    mesh = ProcessMesh([0, 1], dim_names=["x"])
    out_dist_attrs = [
        (mesh, [dist.Partial(dist.ReduceType.kRedSum)]),
    ]

    local_input = paddle.arange(0, 10, dtype='float32')
    local_input = local_input + dist.get_rank()
    input_dist = dist.auto_parallel.api.dtensor_from_local(
        local_input,
        mesh,
        [dist.Shard(0)]
    )

    custom_layer = CustomLayer(out_dist_attrs)
    output_dist = custom_layer(input_dist)
    local_value = custom_layer.local_result

    gathered_values = []
    dist.all_gather(gathered_values, local_value)
    print(f"[Rank 0] local_loss={gathered_values[0]}")
    # [Rank 0] local_loss=1.5
    print(f"[Rank 1] local_loss={gathered_values[1]}")
    # [Rank 1] local_loss=6.0
    print(f"global_loss (distributed)={output_dist}")
    # global_loss (distributed)=7.5


方法
:::::::::

__call__()
'''''''''

执行局部计算的核心方法。该方法会:

1. 将输入的分布式张量转换为本地张量
2. 在本地执行前向计算
3. 将计算结果按照指定的分布策略转换回分布式张量

**参数**

    - **inputs** (Any) - 输入张量,通常是分布式张量
    - **kwargs** (Any) - 额外的关键字参数

**返回**

    按照 out_dist_attrs 指定的分布策略转换后的分布式张量

**使用场景**

LocalLayer 可以用于但不限于以下场景:

1. 带 mask 的 loss 计算:需要在每张卡上独立计算 masked token 的 loss
2. MoE (混合专家模型)相关计算:
  - aux_loss 计算:基于每张卡上专家分配到的局部 token 数进行计算
  - z_loss 计算:对每张卡上的 logits 独立计算 z_loss
  - 张量 reshape 操作:在局部维度上进行 shape 变换
3. 其他需要保持局部计算语义的场景

**注意事项**

1. LocalLayer 的输出必须指定正确的分布策略,以确保结果的正确性
2. 在 forward 方法中编写计算逻辑时,可以像单卡编程一样使用常规的 tensor 操作
3. 局部计算结果会自动根据分布策略进行聚合,无需手动添加通信操作
