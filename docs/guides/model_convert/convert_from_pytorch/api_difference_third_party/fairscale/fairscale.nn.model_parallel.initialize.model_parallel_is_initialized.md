## [组合替代实现]fairscale.nn.model_parallel.initialize.model_parallel_is_initialized

### [fairscale.nn.model_parallel.initialize.model_parallel_is_initialized](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/initialize.py#L119)

```python
fairscale.nn.model_parallel.initialize.model_parallel_is_initialized()
```

返回模型并行初始化设置是否完成; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
fairscale.nn.model_parallel.initialize.model_parallel_is_initialized()

# Paddle 写法
paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None
```
