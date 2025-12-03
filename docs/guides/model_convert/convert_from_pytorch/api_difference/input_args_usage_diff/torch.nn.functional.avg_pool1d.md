## [ 输入参数用法不一致 ]torch.nn.functional.avg_pool1d
### [torch.nn.functional.avg_pool1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool1d.html#torch.nn.functional.avg_pool1d)
```python
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

### [paddle.nn.functional.avg_pool1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/avg_pool1d_cn.html#avg-pool1d)
```python
paddle.nn.functional.avg_pool1d(x, kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None)
```

其中 PyTorch 与 Paddle 参数不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  kernel_size    |  kernel_size    | 池化核的尺寸大小。               |
|  stride           |     stride           | 池化操作步长。             |
|  padding              |  padding   | 池化补零的方式。               |
|  ceil_mode              |  ceil_mode   | 是否用 `ceil` 函数计算输出的 height 和 width，如果设置为 `False`，则使用 `floor` 函数来计算，默认为 `False`。             |
|  count_include_pad            |  exclusive             | 是否用额外 padding 的值计算平均池化结果，PyTorch 与 Paddle 的功能相反，需要转写。  |


### 转写示例
#### count_include_pad：是否用额外 padding 的值计算平均池化结果
```python
# PyTorch 写法
torch.nn.functional.avg_pool1d(input=input, kernel_size=2, stride=2, padding=1, ceil_mode=True, count_include_pad=False)

# Paddle 写法
paddle.nn.functional.avg_pool1d(x=input, kernel_size=2, stride=2, padding=1, ceil_mode=True, exclusive=True)
```
