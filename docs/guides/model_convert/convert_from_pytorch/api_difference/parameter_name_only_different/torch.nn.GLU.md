## [仅参数名不一致]torch.nn.GLU

### [torch.nn.GLU](https://pytorch.org/docs/stable/generated/torch.nn.GLU.html#torch.nn.GLU)

```python
torch.nn.GLU(dim=-1)
```

### [paddle.nn.GLU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GLU_cn.html)

```python
paddle.nn.GLU(axis=-1, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| dim   | axis        | GLU 划分输入的轴，仅参数名不一致。                                                                                              |
