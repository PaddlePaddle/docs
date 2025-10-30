## [ 仅 API 调用方式不一致 ]torch.nn.AdaptiveAvgPool3d

### [torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html#torch.nn.AdaptiveAvgPool3d)

```python
torch.nn.AdaptiveAvgPool3d(output_size)
```

### [paddle.nn.AdaptiveAvgPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/AdaptiveAvgPool3D_cn.html#paddle/nn/AdaptiveAvgPool3D_cn#cn-api-paddle-nn-AdaptiveAvgPool3D)

```python
paddle.nn.AdaptiveAvgPool3D(output_size, data_format="NCDHW", name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.AdaptiveAvgPool3d(1)

# Paddle 写法
model = paddle.nn.AdaptiveAvgPool3D(1)
```
