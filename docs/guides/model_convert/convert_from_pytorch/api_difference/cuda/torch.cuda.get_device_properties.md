## [参数完全一致]torch.cuda.get_device_properties

### [torch.cuda.get_device_properties](https://pytorch.org/docs/1.13/generated/torch.cuda.get_device_properties.html#torch.cuda.get_device_properties)

```python
torch.cuda.get_device_properties(device)
```

### [paddle.device.cuda.get_device_properties](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/get_device_properties_cn.html)

```python
paddle.device.cuda.get_device_properties(device)
```

功能一致，参数完全一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | device            | 表示设备、设备 ID 和类似于 gpu:x 的设备名称。如果 device 为空，则 device 为当前的设备。默认值为 None。|
