## [ 仅 API 调用方式不一致 ]torch.nn.AdaptiveAvgPool2d

### [torch.nn.AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d)

```python
torch.nn.AdaptiveAvgPool2d(output_size)
```

### [paddle.nn.AdaptiveAvgPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/AdaptiveAvgPool2D_cn.html#paddle.nn.AdaptiveAvgPool2D)

```python
paddle.nn.AdaptiveAvgPool2D(output_size, data_format="NCHW", name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.AdaptiveAvgPool2d(5)

# Paddle 写法
model = paddle.nn.AdaptiveAvgPool2D(5)
```
