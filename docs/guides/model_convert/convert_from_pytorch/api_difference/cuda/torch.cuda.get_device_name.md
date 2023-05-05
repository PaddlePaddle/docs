## [部分参数不一致]torch.cuda.get_device_name

### [torch.cuda.get_device_name](https://pytorch.org/docs/1.13/generated/torch.cuda.get_device_name.html)

```python
    torch.cuda.get_device_name(device=None)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/get_device_properties_cn.html#get-device-properties)

```python
    paddle.device.cuda.get_device_properties(device)
```

### 一致的参数
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | device | torch的device参数类型为torch.device 或 int 或 str。paddle的id为paddle.CUDAPlace 或 int 或 str。 |

### 不一致的参数
两者返回类型不一致。paddle返回数据包括标识设备的 ASCII 字符串、设备计算能力的主版本号以及次版本号、全局显存总量、设备上多处理器的数量，其中name属性内容与torch的返回内容一致。

```python
    # pytorch
    torch.cuda.get_device_name()

    # paddle
    paddle.device.cuda.get_device_properties()
```