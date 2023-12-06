## [组合替代实现]torch.nn.Module.get_buffer

### [torch.nn.Module.get_buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_buffer)

```python
torch.nn.Module.get_buffer(target)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
module.get_buffer("target")

# Paddle 写法
getattr(module, "target")
```
