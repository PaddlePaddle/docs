## [ 输入参数类型不一致 ]torch.cuda.get_device_properties

### [torch.cuda.get_device_properties](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_properties.html#torch.cuda.get_device_properties)

```python
torch.cuda.get_device_properties(device)
```

### [paddle.device.cuda.get_device_properties](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/get_device_properties_cn.html)

```python
paddle.device.cuda.get_device_properties(device)
```

两者功能一致但参数类型不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | device            | 表示设备、设备 ID 和类似于 gpu:x 的设备名称。默认值为 None，需要转写。|

### 转写示例
#### device: 设备

```python
# PyTorch 写法
torch.cuda.get_device_properties('cuda:0')

# Paddle 写法
paddle.device.cuda.get_device_properties('gpu:0')

# PyTorch 写法
torch.cuda.get_device_properties(2)

# Paddle 写法
paddle.device.cuda.get_device_properties('gpu:2')

# PyTorch 写法
torch.cuda.get_device_properties()

# Paddle 写法
paddle.device.cuda.get_device_properties()
```
