## [仅 paddle 参数更多]torch.Tensor.corrcoef

### [torch.Tensor.corrcoef](https://pytorch.org/docs/stable/generated/torch.Tensor.corrcoef.html#torch.Tensor.corrcoef)

```python
torch.Tensor.corrcoef()
```

### [paddle.linalg.corrcoef](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/corrcoef_cn.html#paddle.linalg.corrcoef)

```python
paddle.linalg.corrcoef(x, rowvar=True, name=None)
```

仅 paddle 参数更多，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                              |
| ------- | ------------ | ----------------------------------------------------------------- |
| -       | rowvar       | 以行或列作为一个观测变量，Pytorch 无此参数，Paddle 保持默认即可。 |
