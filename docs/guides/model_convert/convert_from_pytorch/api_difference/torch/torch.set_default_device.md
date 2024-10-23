## [组合替代实现] torch.set_default_device
### [torch.set_default_device](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)
```python
torch.set_default_device(device)
```

设置默认设备，Paddle 无此 api，需要组合替代实现。

### 转写示例

```python
# torch 写法
torch.set_default_device(device)

# paddle 写法
paddle.device.set_device(device)
```
