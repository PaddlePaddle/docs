## [ 输入参数类型不一致 ]torch.set_default_device

### [torch.set_default_device](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch-set-default-device)

```python
torch.set_default_device(device)
```

### [paddle.device.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/set_device_cn.html#set-device)

```python
paddle.device.set_device(device)
```

功能一致，参数类型不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ |------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 。PaddlePaddle 支持 str。 |


### 转写示例
#### device: 特定的运行设备

```python
# PyTorch 写法
torch.set_default_device('cuda:0')

# Paddle 写法
paddle.device.set_device('gpu:0')

# PyTorch 写法
torch.set_default_device(2)

# Paddle 写法
paddle.device.set_device('gpu:2')

# PyTorch 写法
torch.set_default_device("cpu")

# Paddle 写法
paddle.device.set_device("cpu")
```
