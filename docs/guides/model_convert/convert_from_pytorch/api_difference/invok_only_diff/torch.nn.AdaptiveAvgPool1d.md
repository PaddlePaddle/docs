## [ 仅 API 调用方式不一致 ]torch.nn.AdaptiveAvgPool1d

### [torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html#torch.nn.AdaptiveAvgPool1d)

```python
torch.nn.AdaptiveAvgPool1d(output_size)
```

### [paddle.nn.AdaptiveAvgPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/AdaptiveAvgPool1D_cn.html#paddle/nn/AdaptiveAvgPool1D_cn#cn-api-paddle-nn-AdaptiveAvgPool1D)

```python
paddle.nn.AdaptiveAvgPool1D(output_size, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.AdaptiveAvgPool1d(5)

# Paddle 写法
model = paddle.nn.AdaptiveAvgPool1D(5)
```
