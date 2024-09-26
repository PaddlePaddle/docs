## [ 输入参数用法不一致 ]torch.cuda.Stream

### [torch.cuda.Stream](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream)

```python
torch.cuda.Stream(device=None, priority=0, **kwargs)
```

### [paddle.device.Stream](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/device/Stream_cn.html#stream)

```python
paddle.device.Stream(device=None, priority=None, blocking=False)
```

两者功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                                                      |
| -------- | ------------ | ----------------------------------------------------------------------------------------- |
| device   | device       | 希望分配 stream 的设备。                                                                  |
| priority | priority     | stream 的优先级，PyTorch 取值范围为-1、0，Paddle 的取值范围为 1、2，需要转写。 |
| - | blocking     | stream 是否同步执行，Paddle 保持默认值即可。 |

### 转写示例

#### priority: stream 的优先级

```python
# PyTorch 写法
high_priority = -1
default_priority = 0
y = torch.cuda.Stream(priority=default_priority)

# Paddle 写法
high_priority = 1
default_priority = 2
y = paddle.device.Stream(priority=default_priority)
```
