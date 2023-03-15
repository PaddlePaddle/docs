## [ 仅参数名不一致 ]torch.Tensor.

### [torch.dot]()

```python
torch.dot(input, other, *, out=None)
```

### [paddle.dot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dot_cn.html#dot)

```python
paddle.dot(x, y, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                        |
| ------------------------ | -------------------- | --------------------------- |
| <center> input </center> | <center> x </center> | 输入 Tensor，仅参数名不同。 |
| <center> other </center> | <center> y </center> | 输入 Tensor，仅参数名不同。 |
