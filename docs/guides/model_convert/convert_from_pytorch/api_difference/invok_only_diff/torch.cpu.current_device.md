## [ 仅 API 调用方式不一致 ]torch.cpu.current_device

### [torch.cpu.current\_device](https://docs.pytorch.org/docs/stable/generated/torch.cpu.current_device.html#torch.cpu.current_device)

```python
torch.cpu.current_device()
```

### [paddle.get_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/get_device_cn.html)

```python
paddle.get_device()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cpu.current_device()

# Paddle 写法
result = paddle.get_device()
```
