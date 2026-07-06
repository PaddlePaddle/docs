## [ torch 参数更多 ]torch.linalg.eigh
### [torch.linalg.eigh](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html#torch.linalg.eigh)
```python
torch.linalg.eigh(input, UPLO='L', *, out=None)
```

### [paddle.linalg.eigh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/eigh_cn.html#paddle.linalg.eigh)
```python
paddle.linalg.eigh(x, UPLO='L', name=None, *, out=None)
```

两者功能一致。

### 参数映射

| PyTorch | PaddlePaddle | 备注                                             |
| ------- | ------------ | ------------------------------------------------ |
| input   | x            | 输入 Tensor，仅参数名不一致。 别名 ``input``。                   |
| UPLO    | UPLO         | 表示计算上三角或者下三角矩阵。                     |
| out     | out          | 表示输出的 Tensor 元组。 |
