## [ 仅参数名不一致 ]torch.Tensor.cross

### [torch.Tensor.cross](https://pytorch.org/docs/stable/generated/torch.Tensor.cross.html?highlight=cross#torch.Tensor.cross)

```python
torch.Tensor.cross(other, dim=None)
```

### [paddle.Tensor.cross](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cross-y-axis-none-name-none)

```python
paddle.Tensor.cross(y, axis=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                   |
| ------- | ------------ | -------------------------------------- |
| other   | y            | 输入 Tensor，仅参数名不一致。            |
| dim     | axis         | 沿此维度进行向量积操作，仅参数名不一致。 |
