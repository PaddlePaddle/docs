## [ paddle 参数更多 ]torch.Tensor.fliplr

### [torch.fliplr](https://pytorch.org/docs/stable/generated/torch.fliplr.html?highlight=fliplr#torch.fliplr)

```python
torch.fliplr(input)
```

### [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flip_cn.html#flip)

```python
paddle.flip(x, axis, name=None)
```

两者功能一致且参数用法一致，paddle 参数更多，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle            | 备注                                                                       |
| ------------------------ | ----------------------- | -------------------------------------------------------------------------- |
| <center> input </center> | <center> x </center>    | 输入 Tensor，pytorch：输入至少是二维。                                     |
| <center> - </center>     | <center> axis </center> | pytorch：沿左/右方向翻转每行中的条目。列被保留，但出 ​​ 现的顺序与以前不同 |
