## [ 仅 API 调用方式不一致 ]torch.backends.cudnn.version

### [torch.backends.cudnn.version](https://pytorch.org/docs/stable/generated/torch.backends.cudnn.html#torch.backends.cudnn.version)

```python
torch.backends.cudnn.version()
```

### [paddle.device.get_cudnn_version](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/get_cudnn_version_cn.html#paddle/device/get_cudnn_version_cn#cn-api-paddle-device-get_cudnn_version)

```python
paddle.device.get_cudnn_version()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.backends.cudnn.version()

# Paddle 写法
result = paddle.device.get_cudnn_version()

```
