## [ 仅参数名不一致 ]torch.Tensor.dist

### [torch.dist](https://pytorch.org/docs/stable/generated/torch.dist.html?highlight=dist#torch.dist)

```python
torch.dist(input, other, p=2)
```

### [paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dist_cn.html#dist)

```python
paddle.dist(x, y, p=2, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                        |
| ------------------------ | -------------------- | --------------------------- |
| <center> input </center> | <center> x </center> | 输入 Tensor，仅参数名不同。 |
|  |
| <center> other </center> | <center> y </center> | 输入 Tensor，仅参数名不同。 |
| <center> p </center>     | <center> p </center> | 需要计算的范数。            |
