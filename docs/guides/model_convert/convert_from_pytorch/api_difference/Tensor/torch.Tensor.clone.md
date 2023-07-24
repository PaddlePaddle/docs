## [ torch 参数更多 ] torch.Tensor.clone

### [torch.Tensor.clone](https://pytorch.org/docs/stable/generated/torch.Tensor.clone.html#torch.Tensor.clone)

```python
torch.Tensor.clone(*, memory_format=torch.preserve_format)
```

### [paddle.Tensor.clone](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#clone)

```python
paddle.Tensor.clone()
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
