## [ Paddle 参数更多 ]torch.nn.functional.lp_pool2d

### [torch.nn.functional.lp_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool2d.html#torch.nn.functional.lp_pool2d)

```python
torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
```

### [paddle.nn.functional.lp_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/lp_pool2d_cn.html#lp-pool2d)
```python
paddle.nn.functional.lp_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, data_format="NCL", name=None)
```

其中 PyTorch 与 Paddle 参数不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。  |
| norm_type     | norm_type    | 幂平均池化的指数，不可以为 0 。 |
| kernel_size   | kernel_size  | 池化核的尺寸大小。|
| stride        | stride       | 池化操作步长。|
| ceil_mode     | ceil_mode    | 是否用 `ceil` 函数计算输出的 height 和 width，如果设置为 `False`，则使用 `floor` 函数来计算，默认为 `False`。|
| -             | padding      | 池化补零的方式。PyTorch 无此参数，Paddle 保持默认即可。|
| -             | data_format  | 输入和输出的数据格式，可以是 `NCHW` 和 `NHWC` 。PyTorch 无此参数，Paddle 保持默认即可。|
