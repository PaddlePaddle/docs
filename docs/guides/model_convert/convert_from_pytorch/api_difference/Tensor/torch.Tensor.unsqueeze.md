## [仅参数名不一致]torch.Tensor.unsqueeze

### [torch.Tensor.unsqueeze](https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html#torch.Tensor.unsqueeze)

```python
torch.Tensor.unsqueeze(dim)
```

### [paddle.Tensor.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#unsqueeze-axis-name-none)

```python
paddle.Tensor.unsqueeze(axis, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                备注                |
| ------- | ------------ | ---------------------------------- |
|   dim   |     axis     | 表示进行运算的轴，仅参数名不一致。 |
