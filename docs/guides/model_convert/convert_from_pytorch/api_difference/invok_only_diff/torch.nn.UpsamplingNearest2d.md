## [ 仅 API 调用方式不一致 ]torch.nn.UpsamplingNearest2d

### [torch.nn.UpsamplingNearest2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.UpsamplingNearest2d.html#torch.nn.UpsamplingNearest2d)

```python
torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)
```

### [paddle.nn.UpsamplingNearest2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/UpsamplingNearest2D_cn.html#paddle.nn.UpsamplingNearest2D)

```python
paddle.nn.UpsamplingNearest2D(size=None, scale_factor=None, data_format='NCHW', name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.UpsamplingNearest2d(scale_factor=2)

# Paddle 写法
model = paddle.nn.UpsamplingNearest2D(scale_factor=2)
```
