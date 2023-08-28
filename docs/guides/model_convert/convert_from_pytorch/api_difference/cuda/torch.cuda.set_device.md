## [参数不一致]torch.cuda.set_device

### [torch.cuda.set_device](https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html#torch.cuda.set_device)

```python
torch.cuda.set_device(device)
```

### [paddle.device.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html)

```python
paddle.device.set_device(device)
```

功能一致，参数类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ |------------------------------------------------|
| device        | device            | PyTorch 支持 torch.device 或 int。PaddlePaddle 支持 str。 |
