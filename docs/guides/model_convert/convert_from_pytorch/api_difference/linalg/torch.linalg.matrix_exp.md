## [ 仅参数名不一致 ]torch.linalg.matrix_exp
### [torch.linalg.matrix_exp](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html#torch.linalg.matrix_exp)

```python
torch.linalg.matrix_exp(A)
```

### [paddle.linalg.matrix_exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/matrix_exp_cn.html)

```python
paddle.linalg.matrix_exp(x, name=None)
```

Pytorch 相比 Paddle 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A          |  x           | 输入的方阵，类型为 Tensor,仅参数名不一致。  |
