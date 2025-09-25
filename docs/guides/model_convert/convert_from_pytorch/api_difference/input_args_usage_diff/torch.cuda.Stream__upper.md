## [ 输入参数用法不一致 ]torch.cuda.Stream

### [torch.cuda.Stream](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream)

```python
torch.cuda.Stream(device=None, priority=0, **kwargs)
```

### [paddle.device.Stream](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/Stream_cn.html#stream)

```python
paddle.device.Stream(device=None, priority=None)
```

两者功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                                                      |
| -------- | ------------ | ----------------------------------------------------------------------------------------- |
| device   | device       | 希望分配 stream 的设备。                                                                  |
| priority | priority     | stream 的优先级，PyTorch 取值范围为-1、0，Paddle 的取值范围为 1、2，需要转写。 |

### 转写示例

#### priority: stream 的优先级

```python
# PyTorch 写法
torch.cuda.Stream(priority=0)

# Paddle 写法
paddle.device.Stream(priority=2)
```

#### device: 希望分配 stream 的设备

```python
# PyTorch 写法
torch.cuda.Stream('cuda:0')

# Paddle 写法
paddle.device.Stream('gpu:0')

# PyTorch 写法
torch.cuda.Stream(2)

# Paddle 写法
paddle.device.Stream(device='gpu:2')
```
