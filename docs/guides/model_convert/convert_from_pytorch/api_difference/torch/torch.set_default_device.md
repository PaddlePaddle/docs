## [ 输入参数类型不一致 ]torch.set_default_device

### [torch.set_default_device](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch-set-default-device)

```python
torch.set_default_device(device)
```

### [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/set_device_cn.html#set-device)

```python
paddle.set_device(device)
```

功能一致，参数类型不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ |------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 。PaddlePaddle 支持 str。 |
