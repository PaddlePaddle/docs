## [ torch 参数更多 ]torch.linalg.cross
### [torch.linalg.cross](https://docs.pytorch.org/docs/stable/generated/torch.linalg.cross.html#torch.linalg.cross)
```python
torch.linalg.cross(input, other, *, dim=- 1, out=None)
```

### [paddle.cross](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cross_cn.html#paddle.cross)
```python
paddle.cross(x, y, axis=-1, name=None, *, out=None)
```

两者功能一致。

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input         | x      | 表示输入的 Tensor ，仅参数名不一致。 别名 ``input``。                        |
| other         | y      | 表示输入的 Tensor ，仅参数名不一致。 别名 ``other``。                        |
| dim       | axis        | 表示进行运算的维度。别名 ``dim``。                       |
| out           | out      | 表示输出的 Tensor。         |
