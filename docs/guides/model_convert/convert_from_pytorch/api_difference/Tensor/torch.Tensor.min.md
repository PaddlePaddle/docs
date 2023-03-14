## [仅参数名称不一致]torch.Tensor.min

### [torch.Tensor.min](https://pytorch.org/docs/stable/generated/torch.Tensor.min.html)

```python
torch.Tensor.min(dim=None, keepdim=False)
```

### [paddle.Tensor.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#min-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.median(axis=None, keepdim=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

| PyTorch                            | PaddlePaddle                       | 备注                                     |
|------------------------------------|------------------------------------|----------------------------------------|
| <font color='red'> dim </font>     | <font color='red'> axis </font>    | 指定进行运算的轴，仅参数名不同。                       |
| <font> keepdim </font> | <font> keepdim </font> | 是否在输出 Tensor 中保留减小的维度。                 |