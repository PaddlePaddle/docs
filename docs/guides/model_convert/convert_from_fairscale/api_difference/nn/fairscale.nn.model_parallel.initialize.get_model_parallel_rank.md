## [无参数]fairscale.nn.model_parallel.initialize.get_model_parallel_rank

### [fairscale.nn.model_parallel.initialize.get_model_parallel_rank](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L155)

```python
fairscale.nn.model_parallel.initialize.get_model_parallel_rank()
```

### [paddle.distributed.get_rank](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_rank_cn.html)

```python
paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
```

两者功能一致，均无参数。

### 转写示例
```python
# PyTorch 写法
fairscale.nn.model_parallel.initialize.get_model_parallel_size()

# Paddle 写法
assert paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None
paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
```
