## [ 参数不一致 ]torch.nn.functional.avg_pool3d

### [torch.nn.functional.avg_pool3d](https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool3d.html#torch.nn.functional.avg_pool3d)

```python
torch.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

### [paddle.nn.functional.avg_pool3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/avg_pool3d_cn.html#avg-pool3d)
```python
paddle.nn.functional.avg_pool3d(x, kernel_size, stride=None, padding=0, ceil_mode=False, exclusive=True, divisor_override=None, data_format='NCDHW', name=None)
```

其中 Pytorch 与 Paddle 参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> kernel_size </font>   | <font color='red'> kernel_size </font>   | 池化核的尺寸大小。               |
| <font color='red'> stride  </font>         |    <font color='red'> stride  </font>         | 池化操作步长。             |
| <font color='red'> padding </font>             | <font color='red'> padding </font>  | 池化补零的方式。               |
| <font color='red'> ceil_mode </font>             | <font color='red'> ceil_mode </font>  | 是否用 `ceil` 函数计算输出的 height 和 width，如果设置为 `False`，则使用 `floor` 函数来计算，默认为 `False`            |
| <font color='red'> divisor_override </font>           | <font color='red'> divisor_override </font>            | 如果指定，它将用作除数，否则根据 `kernel_size` 计算除数。默认 `None`  |
| -           | <font color='red'> data_format </font>            | 输入和输出的数据格式, Pytorch 无此参数  |
| <font color='red'> count_include_pad </font>           | <font color='red'> exclusive </font>            | 是否用额外 padding 的值计算平均池化结果，Pytorch 与 Paddle 的功能相反，需要进行转写  |


### 转写示例
#### count_include_pad：是否用额外 padding 的值计算平均池化结果
```python
# Pytorch 写法
torch.nn.functional.avg_pool3d(input=input, kernel_size=2, stride=2, padding=1, ceil_mode=True, count_include_pad=False)

# Paddle 写法
paddle.nn.functional.avg_pool3d(x=input, kernel_size=2, stride=2, padding=1, ceil_mode=True, exlusive=True)
```
