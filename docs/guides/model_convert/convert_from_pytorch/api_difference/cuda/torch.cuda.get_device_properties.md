## [输入参数用法不一致]torch.cuda.get_device_properties

### [torch.cuda.get_device_properties](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_properties.html#torch.cuda.get_device_properties)

```python
torch.cuda.get_device_properties(device)
```

### [paddle.device.cuda.get_device_properties](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/get_device_properties_cn.html)

```python
paddle.device.cuda.get_device_properties(device)
```

功能一致，输入参数用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | device            | 表示设备、设备 ID 和类似于 gpu:x 的设备名称。如果 device 为空，则 device 为当前的设备。默认值为 None，需要转写。|

### 转写示例
#### device: 设备

```python
# PyTorch 写法
torch.cuda.get_device_properties('cuda:0')

# Paddle 写法
paddle.device.cuda.get_device_properties('gpu:0')

# PyTorch 写法
num=2
torch.cuda.get_device_properties(num)

# Paddle 写法
paddle.device.cuda.get_device_properties(device=f'gpu:{type}')

# PyTorch 写法
torch.cuda.get_device_properties(device=0 if 2 > 1 else 1)

# Paddle 写法
paddle.device.cuda.get_device_properties('gpu:0')
```
