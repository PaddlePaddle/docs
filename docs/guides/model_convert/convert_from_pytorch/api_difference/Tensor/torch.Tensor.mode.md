## [仅参数名称不一致]torch.Tensor.mode

### [torch.Tensor.mode](https://pytorch.org/docs/stable/generated/torch.Tensor.mode.html)

```python
torch.Tensor.nanmean(dim=None, keepdim=False)
```

### [paddle.Tensor.mode](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#mode-axis-1-keepdim-false-name-none)
```python
paddle.Tensor.nanmedian(axis=None, keepdim=True, name=None)
```
两者功能一致且参数用法一致，仅参数名不同，具体如下：

| PyTorch                            | PaddlePaddle                       | 备注                                     |
|------------------------------------|------------------------------------|----------------------------------------|
| <font color='red'> dim </font>     | <font color='red'> axis </font>    | 指定进行运算的轴，仅参数名不同。                       |
| <font color='red'> keepdim </font> | <font color='red'> keepdim </font> | 是否在输出 Tensor 中保留减小的维度。                 |