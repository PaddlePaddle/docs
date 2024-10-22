## [组合替代实现] torch.cuda.device_of

### [torch.cuda.device_of](https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html#torch.cuda.device_of)
```python
torch.cuda.device_of(obj)
```

获取张量所在的设备，Paddle 无此 api，需要组合实现
可以通过`tensor.place`来获取张量所在的设备信息

### 转写示例
```python
# torch 写法
device = torch.cuda.device_of(tensor)

# paddle 写法
device = tensor.place
```
