## [参数完全一致]fairscale.nn.model_parallel.initialize.get_model_parallel_world_size

### [fairscale.nn.model_parallel.initialize.get_model_parallel_world_size](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L173)

```python
fairscale.nn.model_parallel.initialize.get_model_parallel_size(group)
```

### [paddle.distributed.get_world_size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_world_size_cn.html)

```python
paddle.distributed.get_world_size(group=None)
```


功能一致, fairscale 要求 group 参数非 None，paddle 额外支持参数为 None 的情形，具体如下：

### 参数映射

| fairscale | PaddlePaddle | 备注     |
| --------- | ------------ | -------- |
| group     | group        | 进程组。 |
