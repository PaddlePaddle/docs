## [组合替代实现] torch.get_default_device

### [torch.get_default_device](https://pytorch.org/docs/stable/generated/torch.get_default_device.html)
```python
torch.get_default_device()
```

获取默认的设备，Paddle 无此 api， 需要组合实现

### 转写示例
```python
# torch 写法
device = torch.get_default_device()

# paddle 写法
device = paddle.device.get_device()
```
