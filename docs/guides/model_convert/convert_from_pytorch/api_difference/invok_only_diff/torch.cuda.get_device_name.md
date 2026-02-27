## [ 仅 API 调用方式不一致 ]torch.cuda.get_device_name

### [torch.cuda.get\_device\_name](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_name.html#torch.cuda.get_device_name)

```python
torch.cuda.get_device_name(device)
```

### [paddle.device.cuda.get\_device\_name](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/get_device_name_cn.html#paddle.device.cuda.get_device_name)

```python
paddle.device.cuda.get_device_name(device)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.get_device_name(current_device)

# Paddle 写法
result = paddle.device.cuda.get_device_name(current_device)
```
