## [ 仅 API 调用方式不一致 ]torch.nn.MaxUnpool2d

### [torch.nn.MaxUnpool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html#torch.nn.MaxUnpool2d)

```python
torch.nn.MaxUnpool2d(kernel_size, stride, padding)
```

### [paddle.nn.MaxUnPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MaxUnPool2D_cn.html#paddle/nn/MaxUnPool2D_cn#cn-api-paddle-nn-MaxUnPool2D)

```python
paddle.nn.MaxUnPool2D(kernel_size, stride, padding, data_format, output_size, name)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
unpool = torch.nn.MaxUnpool2d(2, stride=2)

# Paddle 写法
unpool = paddle.nn.MaxUnPool2D(2, stride=2)
```
