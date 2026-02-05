## [ 输入参数用法不一致 ]torch.nn.functional.avg_pool2d
### [torch.nn.functional.avg\_pool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html#torch.nn.functional.avg_pool2d)
```python
torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

### [paddle.nn.functional.avg\_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/avg_pool2d_cn.html#paddle.nn.functional.avg_pool2d)
```python
paddle.nn.functional.avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, exclusive=True, divisor_override=None, data_format='NCHW', name=None)
```

其中 PyTorch 与 Paddle 参数不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  kernel_size    |  kernel_size    | 池化核的尺寸大小。               |
|  stride           |     stride           | 池化操作步长。             |
|  padding              |  padding   | 池化补零的方式。               |
|  ceil_mode              |  ceil_mode   | 是否用 `ceil` 函数计算输出的 height 和 width，如果设置为 `False`，则使用 `floor` 函数来计算，默认为 `False`。            |
|  divisor_override            |  divisor_override             | 如果指定，它将用作除数，否则根据 `kernel_size` 计算除数。默认 `None`。  |
| -           |  data_format             | 输入和输出的数据格式, PyTorch 无此参数， 保持默认即可。 |
|  count_include_pad            |  exclusive             | 是否用额外 padding 的值计算平均池化结果，PyTorch 与 Paddle 的功能相反，需要转写。  |


### 转写示例
#### count_include_pad：是否用额外 padding 的值计算平均池化结果
```python
# PyTorch 写法
torch.nn.functional.avg_pool2d(input=input, kernel_size=2, stride=2, padding=1, ceil_mode=True, count_include_pad=False)

# Paddle 写法
paddle.nn.AvgPool2D(x=input, kernel_size=2, stride=2, padding=1, ceil_mode=True, exclusive=True)
```
