## [ 仅参数名不一致 ]torch.Tensor.flatten

### [torch.flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html?highlight=flatten#torch.flatten)

```python
torch.flatten(input, start_dim=0, end_dim=- 1)
```

### [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flatten_cn.html#flatten)

```python
paddle.flatten(x, start_axis=0, stop_axis=- 1, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                      | PaddlePaddle                  | 备注                           |
| ---------------------------- | ----------------------------- | ------------------------------ |
| <center> input </center>     | <center> x </center>          | 输入 Tensor，仅参数名不同。    |
| <center> start_dim </center> | <center> start_axis </center> | 展开的起始维度，仅参数名不同。 |
| <center> end_dim </center>   | <center> stop_axis</center>   | 展开的结束维度，仅参数名不同。 |
