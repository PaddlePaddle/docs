## [ 输入参数类型不一致 ]torch.cuda.set_device

### [torch.cuda.set_device](https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html#torch.cuda.set_device)

```python
torch.cuda.set_device(device)
```

### [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html)

```python
paddle.set_device(device)
```

功能一致，参数类型不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ |------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 或 int。PaddlePaddle 支持 str。 |


### 转写示例
#### device: 特定的运行设备

```python
# PyTorch 写法
torch.cuda.set_device('cuda:0')

# Paddle 写法
paddle.set_device('gpu:0')

# PyTorch 写法
torch.cuda.set_device(2)

# Paddle 写法
paddle.set_device('gpu:2')
```
