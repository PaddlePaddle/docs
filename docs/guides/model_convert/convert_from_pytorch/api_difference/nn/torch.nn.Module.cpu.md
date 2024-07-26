## [ paddle 参数更多 ]torch.nn.Module.cpu

### [torch.nn.Module.cpu](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.cpu)

```python
torch.nn.Module.cpu()
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to(device=None, dtype=None, blocking=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                              |
| ------- | ------------ | --------------------------------- |
| -       | device       | 移动到的设备，PyTorch 无此参数，Paddle 设置为 "cpu"。 |
| -       | dtype        | 数据类型，PyTorch 无此参数，Paddle 保持默认即可。     |
| -       | blocking     | 是否阻塞，PyTorch 无此参数，Paddle 保持默认即可。     |
