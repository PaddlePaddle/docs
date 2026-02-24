## [ 仅 API 调用方式不一致 ]torch.nn.ZeroPad2d

### [torch.nn.ZeroPad2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html#torch.nn.ZeroPad2d)

```python
torch.nn.ZeroPad2d(padding)
```

### [paddle.nn.ZeroPad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ZeroPad2D_cn.html#paddle.nn.ZeroPad2D)

```python
paddle.nn.ZeroPad2D(padding, data_format='NCHW', name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.ZeroPad2d(1)

# Paddle 写法
model = paddle.nn.ZeroPad2D(1)
```
