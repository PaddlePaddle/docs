## [ 参数用法不一致 ]torch.Tensor.equal

### [torch.Tensor.equal](https://pytorch.org/docs/stable/generated/torch.Tensor.equal.html?highlight=equal#torch.Tensor.equal)

```python
Tensor.equal(other)
```

### [paddle.Tensor.equal_all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#equal-all-y-name-none)

```python
Tensor.equal_all(y, name=None)
```

两者功能一致，但参数用法不一致，其中 torch 返回值是 `bool`，paddle 返回值是数据类型为 `bool` 的 Tensor，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| other   | y            | 输入 Tensor，仅参数名不同。 |
