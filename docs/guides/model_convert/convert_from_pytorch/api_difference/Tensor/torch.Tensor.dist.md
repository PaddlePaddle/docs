## [ 仅参数名不一致 ]torch.Tensor.dist

### [torch.Tensor.dist](https://pytorch.org/docs/stable/generated/torch.Tensor.dist.html?highlight=dist#torch.Tensor.dist)

```python
torch.Tensor.dist(other, p=2)
```

### [paddle.Tensor.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#dist-y-p-2)

```python
paddle.Tensor.dist(y, p=2)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| other   | y            | 输入 Tensor，仅参数名不同。 |
| p       | p            | 需要计算的范数。            |
