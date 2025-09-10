## [ 输入参数类型不一致 ]torch.cpu.set_device

### [torch.cpu.set_device](https://pytorch.org/docs/stable/generated/torch.cpu.set_device.html)

```python
torch.cpu.set_device(device)
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
torch.cpu.set_device('cpu:0')

# Paddle 写法
paddle.device.set_device('cpu')
```
