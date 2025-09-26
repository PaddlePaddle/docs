## [ 参数完全一致 ]torch.cuda.get_device_name

### [torch.cuda.get_device_name](https://pytorch.org/docs/stable/generated/torch.cuda.get_device_name.html)

```python
torch.cuda.get_device_name(device=None)
```

### [paddle.device.cuda.get_device_name](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/get_device_name_cn.html)

```python
paddle.device.cuda.get_device_name(device=None)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                            |
|---------|--------------|-----------------------------------------------------------------------------------------------|
| device  | device       | torch 的 device 参数类型为 torch.device 或 int 或 str。paddle 的 device 为 paddle.CUDAPlace 或 int 或 str。 |
