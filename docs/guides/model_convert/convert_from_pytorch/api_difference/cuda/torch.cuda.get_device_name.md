## [参数不一致]torch.cuda.get_device_name

### [torch.cuda.get_device_name](https://pytorch.org/docs/1.13/generated/torch.cuda.get_device_name.html)

```python
torch.cuda.get_device_name(device=None)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/get_device_properties_cn.html#get-device-properties)

```python
paddle.device.cuda.get_device_properties(device)
```

两者的返回参数不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | device | torch 的 device 参数类型为 torch.device 或 int 或 str。paddle 的 id 为 paddle.CUDAPlace 或 int 或 str。 |
| Returns | Returns | 两者返回类型不一致。 paddle 的返回数据中 name 属性内容与 torch 的返回内容一致。需要转写。|

### 转写示例
```python
# pytorch
torch.cuda.get_device_name()

# paddle
paddle.device.cuda.get_device_properties().name
```
