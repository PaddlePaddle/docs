## [ paddle 参数更多 ]torch.Tensor.flipud

### [torch.flipud](https://pytorch.org/docs/stable/generated/torch.flipud.html?highlight=flipud#torch.flipud)

```python
torch.flipud(input)
```

### [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flip_cn.html#flip)

```python
paddle.flip(x, axis, name=None)
```

两者功能一致且参数用法一致，paddle 参数更多，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle            | 备注                                         |
| ------------------------ | ----------------------- | -------------------------------------------- |
| <center> input </center> | <center> x </center>    | 输入 Tensor。pytorch：张量至少是一维的       |
| <center> - </center>     | <center> axis </center> | pytorch：向上/向下翻转张量，返回一个新的张量 |
