## [输入参数类型不一致]torch.cuda.current_stream

### [torch.cuda.current_stream](https://pytorch.org/docs/stable/generated/torch.cuda.current_stream.html#torch.cuda.current_stream)

```python
torch.cuda.current_stream(device=None)
```

### [paddle.device.current_stream](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/current_stream_cn.html)

```python
paddle.device.current_stream(device=None)
```

功能一致，参数类型不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device        | device            | 表示希望获取 stream 的设备或者设备 ID。如果为 None，则为当前的设备。默认值为 None，需要转写。                                     |

### 转写示例
#### device: 特定的运行设备

```python
# PyTorch 写法
torch.cuda.current_stream('cuda:0')

# Paddle 写法
paddle.device.current_stream('gpu:0')

# PyTorch 写法
torch.cuda.current_stream(2)

# Paddle 写法
paddle.device.current_stream('gpu:2')

# PyTorch 写法
torch.cuda.current_stream()

# Paddle 写法
paddle.device.current_stream()
```
