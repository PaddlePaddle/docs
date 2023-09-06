## [torch 参数更多]torch.Tensor.type

### [torch.Tensor.type](https://pytorch.org/docs/stable/generated/torch.Tensor.type.html#torch.Tensor.type)

```python
torch.Tensor.type(dtype=None, non_blocking=False, **kwargs)
```

### [paddle.Tensor.astype](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#astype-dtype)

```python
paddle.Tensor.astype(dtype)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| dtype        | dtype        | 转换后的 dtype。                                             |
| non_blocking | -            | 控制 cpu 和 gpu 数据的异步复制，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
