## [组合替代实现]torch.cuda.device_of

### [torch.cuda.device_of](https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html)

```python
torch.cuda.device_of(obj)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.cuda.device_of(obj)

# Paddle 写法
device_id = obj.place.gpu_device_id()
paddle.set_device(f"gpu:{device_id}")
paddle.device.get_device()
```
