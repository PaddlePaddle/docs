### [ 无参数 ] torch.cuda.seed

### [torch.cuda.seed_all](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_name.html#torch-cuda-get-device-name)

```python
torch.cuda.get_device_name()
```

### [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/get_device_name_cn.html#get-device-name)

```python
paddle.device.cuda.get_device_name()
```

Paddle 相比 Pytorch 参数更多

### 参数映射
| PyTorch | Paddle | 备注                                                     |
|---------|--------| -------------------------------------------------------- |
| device  | device   | 希望获取名称的设备或者设备 ID。如果 device 为 None（默认），则为当前的设备 |
