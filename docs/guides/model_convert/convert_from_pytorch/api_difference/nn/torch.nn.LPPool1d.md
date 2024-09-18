## [ Paddle 参数更多 ]torch.nn.LPPool1d

### [torch.nn.LPPool1d](https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#lppool1d)

```python
torch.nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False)
```

### [paddle.nn.LPPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/LPPool1D_cn.html#lppool1d)
```python
paddle.nn.LPPool1D(norm_type, kernel_size, stride=None, padding=0, ceil_mode=False, data_format='NCL', name=None)
```

其中 PyTorch 与 Paddle 参数不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> norm_type </font> | <font color='red'> norm_type </font> | 幂平均池化的指数，不可以为 0 。 |
| <font color='red'> kernel_size </font>   | <font color='red'> kernel_size </font>   | 池化核的尺寸大小。               |
| <font color='red'> stride  </font>         |    <font color='red'> stride  </font>         | 池化操作步长。             |
| <font color='red'> ceil_mode </font>             | <font color='red'> ceil_mode </font>  | 是否用 `ceil` 函数计算输出的 height 和 width，如果设置为 `False`，则使用 `floor` 函数来计算，默认为 `False`             |
