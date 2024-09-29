## [torch 参数更多]fairscale.nn.model_parallel.layers.RowParallelLinear

### [fairscale.nn.model_parallel.layers.RowParallelLinear](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L299)

```python
fairscale.nn.model_parallel.layers.RowParallelLinear(in_features: int, out_features: int, bias: bool = True, input_is_parallel: bool = False, init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_, stride: int = 1, keep_master_weight_for_test: bool = False)
```

### [paddle.distributed.fleet.meta_parallel.RowParallelLinear](https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L291)

```python
paddle.distributed.fleet.meta_parallel.RowParallelLinear(in_features, out_features, weight_attr=None, has_bias=True, input_is_parallel=False, fuse_matmul_bias=False, mp_group=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| fairscale                   | PaddlePaddle     | 备注      |
| ----------------------------| ---------------- | -------- |
| in_features                 | in_features      | 输入特征数。 |
| out_features                | out_features     | 输出特征数。 |
| bias                        | has_bias         | 是否增加 bias。 |
| input_is_parallel           | input_is_parallel| 输入是否在 GPUs 上进行过分割，如果是就不再分割。 |
| init_method                 |-                 | 参数初始化方法，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
|-                            | weight_attr      | 网络层参数属性，PyTorch 无此参数，Paddle 保持默认即可。|
| stride                      |-                 | 线性层切分后参数块的 stride, 用于特殊的存储格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| keep_master_weight_for_test |-                 | 返回主参数用于测试，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
|-                            | fuse_matmul_bias | 是否融合矩阵乘和加 bias 操作，PyTorch 无此参数，Paddle 保持默认即可。 |
|-                            | mp_group         | 模型并行组，PyTorch 无此参数，Paddle 保持默认即可。|
|-                            | name             | 网络层名称，PyTorch 无此参数，Paddle 保持默认即可。|
