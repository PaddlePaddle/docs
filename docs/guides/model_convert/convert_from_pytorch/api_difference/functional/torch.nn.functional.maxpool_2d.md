## torch.nn.functional.max_pool2d

### [torch.nn.functional.max_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html?highlight=max_pool2d#torch.nn.functional.max_pool2d)

```python
torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
```

### [paddle.nn.functional.max_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/max_pool2d_cn.html#max-pool2d)

```python
paddle.nn.functional.max_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, return_mask=False, data_format='NCHW', name=None)
```

功能一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor                                      |
| kernel_size   | kernel_size  | 池化窗口的大小                                     |
| stride        | stride       | 池化窗口滑动步长                                     |
| padding       | padding      | 池化的填充                                     |
| ceil_mode     | ceil_mode    | 是否用 ceil 函数计算输出的高和宽                                     |
| return_indices| return_mask  | 是否返回最大值的索引                                                |
| dilation      | -            | 空洞系数，暂不支持 dilation 不等于 1 时的转写                                       |
