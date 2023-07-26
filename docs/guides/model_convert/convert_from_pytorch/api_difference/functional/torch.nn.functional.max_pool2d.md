## [ torch 参数更多 ]torch.nn.functional.max_pool2d

### [torch.nn.functional.max_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d)

```python
torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
```

### [paddle.nn.functional.max_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/max_pool2d_cn.html)

```python
paddle.nn.functional.max_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, return_mask=False, data_format='NCHW', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                          |
| -------------- | ------------ | ------------------------------------------------------------- |
| input          | x            | 输入的 Tensor，仅参数名不一致。                               |
| kernel_size    | kernel_size  | 池化核的尺寸大小。                                            |
| stride         | stride       | 池化操作步长。                                                |
| padding        | padding      | 池化补零的方式。                                              |
| dilation       | -            | 带有滑动窗口元素间的 stride，Paddle 无此参数，暂无转写方式。  |
| ceil_mode      | ceil_mode    | 是否用 ceil 函数计算输出的 height 和 width。                  |
| return_indices | return_mask  | 是否返回最大值的索引，仅参数名不一致。                        |
| -              | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。 |
