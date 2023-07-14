## [仅参数名不一致]torch.nn.GLU

### [torch.nn.GLU](https://pytorch.org/docs/1.13/generated/torch.nn.GLU.html#torch.nn.GLU)

```python
torch.nn.GLU(dim=-1)
```

### [paddle.nn.functional.glu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/glu_cn.html)

```python
paddle.nn.functional.glu(x, axis=-1, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                   |
| ------- | ------------ | -------------------------------------- |
| dim     | axis         | 沿着该轴将输入二等分，仅参数名不一致。 |
