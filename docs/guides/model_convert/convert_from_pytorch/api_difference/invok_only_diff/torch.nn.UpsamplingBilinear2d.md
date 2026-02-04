## [ 仅 API 调用方式不一致 ]torch.nn.UpsamplingBilinear2d

### [torch.nn.UpsamplingBilinear2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.UpsamplingBilinear2d.html#torch.nn.UpsamplingBilinear2d)

```python
torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)
```

### [paddle.nn.UpsamplingBilinear2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/UpsamplingBilinear2D_cn.html#paddle.nn.UpsamplingBilinear2D)

```python
paddle.nn.UpsamplingBilinear2D(size=None, scale_factor=None, data_format='NCHW', name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.UpsamplingBilinear2d(scale_factor=2)

# Paddle 写法
model = paddle.nn.UpsamplingBilinear2D(scale_factor=2)
```
