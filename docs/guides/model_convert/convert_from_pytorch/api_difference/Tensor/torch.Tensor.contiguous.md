## [ torch 参数更多 ] torch.Tensor.contiguous

### [torch.Tensor.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)

```python
torch.Tensor.contiguous(memory_format=torch.contiguous_format)
```

### [paddle.Tensor.contiguous](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#contiguous)

```python
paddle.Tensor.contiguous()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| memory_format | - |表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
