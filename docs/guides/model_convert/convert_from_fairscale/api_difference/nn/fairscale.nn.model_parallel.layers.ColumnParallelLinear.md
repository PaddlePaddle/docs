## [参数不一致]fairscale.nn.model_parallel.layers.ColumnParallelLinear

### [fairscale.nn.model_parallel.layers.ColumnParallelLinear](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L218)

```python
fairscale.nn.model_parallel.initialize.ColumnParallelLinear(in_features,out_features,bias,gather_output,init_method,stride,keep_master_weight_for_test)
```
### [paddle.distributed.meta_parallel.parallel_layers.mp_layers.ColumnParallelLinear](https://github.com/PaddlePaddle/Paddle/blob/016766cc89fabc10181453ce70b701dd8ed019f6/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L153)

```python
paddle.distributed.meta_parallel.parallel_layers.mp_layers.ColumnParallelLinear(in_features,out_features,weight_attr,has_bias,gather_output,fuse_matmul_bias,mp_group,name)
```

两者功能大体一致，参数不一致。

### 参数映射

| fairscale | PaddlePaddle | 备注     |
| --------- | ------------ | -------- |
| in_features | in_features| 输入特征数 |
| out_features |out_features |输出特征数|
| bias |has_bias | 是否增加 bias |
| gather_output |gather_output | 是否对每个 rank 的输出 allgather |
| init_method | | 参数初始化方法|
|             |weight_attr | 网络层参数属性|
| stride | | 线性层滑动步长 |
| keep_master_weight_for_test | | 返回主参数用于测试 |
|  |fuse_matmul_bias | 是否融合矩阵乘和加 bias 操作 |
|  | mp_group| 向量并行组|
|  | name| 网络层名称|

### 转写示例

```python
# Pytorch 写法
fairscale.nn.model_parallel.initialize.ColumnParallelLinear(in_features=in_features,
    out_features=out_features,bias=False,gather_out=False)

# Paddle 写法
paddle.distributed.meta_parallel.parallel_layers.mp_layers.ColumnParallelLinear(in_features=in_features,
    out_features=in_features,has_bias=False, gather_output=False)

# Pytorch 写法
fairscale.nn.model_parallel.initialize.ColumnParallelLinear(in_features=in_features,
    out_features=out_features)

# Paddle 写法
paddle.distributed.meta_parallel.parallel_layers.mp_layers.ColumnParallelLinear(in_features=in_features,
    out_features=in_features,has_bias=True)
```
