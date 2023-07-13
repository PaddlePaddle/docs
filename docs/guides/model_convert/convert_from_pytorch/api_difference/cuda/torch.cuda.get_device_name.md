## [ 参数不一致 ]torch.cuda.get_device_name

### [torch.cuda.get_device_name](https://pytorch.org/docs/1.13/generated/torch.cuda.get_device_name.html)

```python
torch.cuda.get_device_name(device=None)
```

### [paddle.device.cuda.get_device_properties](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/get_device_properties_cn.html#get-device-properties)

```python
paddle.device.cuda.get_device_properties(device)
```

两者的返回参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | device | torch 的 device 参数类型为 torch.device 或 int 或 str。paddle 的 device 为 paddle.CUDAPlace 或 int 或 str。 |
| 返回值 | 返回值 | 两者返回类型不一致。torch 返回字符串，paddle 返回包含设备多个属性的数据结构，对其取 name 属性即可。需要转写。|

### 转写示例
#### 返回值
```python
# pytorch
y = torch.cuda.get_device_name()

# paddle
y = paddle.device.cuda.get_device_properties().name
```
