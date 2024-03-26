## [参数完全一致]fairscale.nn.model_parallel.initialize.initialize_model_parallel

### [fairscale.nn.model_parallel.initialize.initialize_model_parallel](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L41)

```python
fairscale.nn.model_parallel.initialize.initialize_model_parallel()
```

对模型并行设置进行初始化; Paddle 无此 API，需要组合实现。

### 参数映射

| fairscale | PaddlePaddle | 备注     |
| --------- | ------------ | -------- |
| model_parallel_size_ | | 模型并行规模 |
| pipeline_length | | 流水线并行规模 |
| model_parallel_backend | | 模型并行通信后端 |
| pipeline_backend | | 流水线并行通信后端 |
| ddp_backend     | | 数据并行通信后端|

### 转写示例

```python
# Pytorch 写法
fairscale.nn.model_parallel.initialize.initialize_model_parallel(model_parallel_size_=model_parallel_size_,pipeline_length=pipeline_length)

# Paddle 写法
world_size = paddle.distributed.get_world_size()
rank = paddle.distributed.get_rank()
model_parallel_size = int(min(world_size,model_parallel_size_))
data_parallel_size = int(world_size/ (model_parallel_size * pipeline_length))
Strategy = paddle.distributed.fleet.DistributedStrategy()
Strategy_dict = dict()
Strategy_dict["dp_degree"] = data_parallel_size
Strategy_dict["mp_degree"] = model_parallel_size
Strategy_dict["pp_degree"] = pipeline_length
Strategy.hybrid_configs = Strategy_dict
paddle.distributed.fleet.init(is_collective=True, strategy=Strategy)
```
