## [仅参数名不一致]torch.Tensor.mean

### [torch.Tensor.mean](https://pytorch.org/docs/stable/generated/torch.Tensor.mean.html#torch.Tensor.mean)

```python
torch.Tensor.mean(dim=None, keepdim=False, *, dtype=None)
```

### [paddle.Tensor.mean](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#mean-axis-none-keepdim-false-name-none)
```python
paddle.Tensor.mean(axis=None, keepdim=False, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下:

| PyTorch                            | PaddlePaddle                       | 备注                                     |
|------------------------------------|------------------------------------|----------------------------------------|
| <font color='red'> dim </font>     | <font color='red'> axis </font>    | 指定进行运算的轴，仅参数名不同。                       |
| <font color='red'> keepdim </font> | <font color='red'> keepdim </font> | 是否在输出 Tensor 中保留减小的维度。                 |
| <font color='red'> dtpye </font>   | <font color='red'> - </font>       | 输出Tensor的数据类型，Paddle无此参数，Pytorch保持默认即可 |
