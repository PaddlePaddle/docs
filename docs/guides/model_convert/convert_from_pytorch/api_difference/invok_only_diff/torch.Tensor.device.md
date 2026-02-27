## [ 仅 API 调用方式不一致 ]torch.Tensor.device

### [torch.Tensor.device](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.device.html#torch.Tensor.device)

```python
torch.Tensor.device
```

### [paddle.Tensor.place](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#place)

```python
paddle.Tensor.place
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = src.device

# Paddle 写法
result = src.place
```
