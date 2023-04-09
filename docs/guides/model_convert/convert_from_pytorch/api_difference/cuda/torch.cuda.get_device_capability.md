## [参数完全一致]torch.cuda.get_device_capability

### [torch.cuda.get_device_capability](https://pytorch.org/docs/1.13/generated/torch.cuda.get_device_capability.html#torch.cuda.get_device_capability)

```python
torch.cuda.get_device_capability(device=None)
```

### [paddle.device.cuda.get_device_capability](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/get_device_capability_cn.html)

```python
paddle.device.cuda.get_device_capability(device=None)
```

功能一致，参数完全一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | device            | 表示希望获取计算能力的设备或者设备 ID。如果 device 为 None（默认），则为当前的设备。 |
