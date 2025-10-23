## [ 仅参数名不一致 ]torch.Tensor.hardshrink

### [torch.Tensor.hardshrink](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.hardshrink)

```python
torch.Tensor.hardshrink(lambd=0.5)
```

### [paddle.nn.functional.hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hardshrink_cn.html#paddle/nn/functional/hardshrink_cn#cn-api-paddle-nn-functional-hardshrink)

```python
paddle.nn.functional.hardshrink(x, threshold=0.5, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| lambd   | threshold     | 阈值，仅参数名不一致，功能一致 |
