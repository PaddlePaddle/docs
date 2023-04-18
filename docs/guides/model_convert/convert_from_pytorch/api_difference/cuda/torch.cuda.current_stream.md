## [参数完全一致]torch.cuda.current_stream

### [torch.cuda.current_stream](https://pytorch.org/docs/1.13/generated/torch.cuda.current_stream.html#torch.cuda.current_stream)

```python
torch.cuda.current_stream(device=None)
```

### [paddle.device.cuda.current_stream](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/current_stream_cn.html)

```python
paddle.device.cuda.current_stream(device=None)
```

功能一致，参数完全一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | device            | 表示希望获取 stream 的设备或者设备 ID。如果为 None，则为当前的设备。默认值为 None。                                     |
