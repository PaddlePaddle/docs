## [ paddle 参数更多 ]torch.Tensor.corrcoef

### [torch.Tensor.corrcoef](https://pytorch.org/docs/stable/generated/torch.Tensor.corrcoef.html#torch.Tensor.corrcoef)

```python
torch.Tensor.corrcoef()
```

### [paddle.Tensor.corrcoef]()

```python
paddle.Tensor.corrcoef(rowvar=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                              |
| ------- | ------------ | ----------------------------------------------------------------- |
| -       | rowvar       | 以行或列作为一个观测变量，PyTorch 无此参数，Paddle 保持默认即可。 |
