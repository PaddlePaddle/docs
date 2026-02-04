## [ 仅 API 调用方式不一致 ]torch.nn.MaxUnpool1d

### [torch.nn.MaxUnpool1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxUnpool1d.html#torch.nn.MaxUnpool1d)

```python
torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
```

### [paddle.nn.MaxUnPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MaxUnPool1D_cn.html#paddle.nn.MaxUnPool1D)

```python
paddle.nn.MaxUnPool1D(kernel_size, stride=None, padding=0, data_format='NCL', output_size=None, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
unpool = torch.nn.MaxUnpool1d(2, 2)
# Paddle 写法)
unpool = paddle.nn.MaxUnPool1D(2, 2)
```
