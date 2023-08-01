## [ 仅参数名不一致 ] torch.Tensor.outer

### [torch.Tensor.outer](https://pytorch.org/docs/stable/generated/torch.Tensor.outer.html?highlight=outer#torch.Tensor.outer)

```python
torch.Tensor.outer(vec2)
```

### [paddle.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html)

```python
paddle.outer(x, y)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                 |
| ------- | ------------ | -------------------------------------------------------------------- |
| vec2    | y            | 如果类型是多维 `Tensor`，其数据类型应该和 `x` 相同，仅参数名不一致。 |
