## [组合替代实现]torch.nn.Module.get_parameter

### [torch.nn.Module.get_parameter](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)

```python
torch.nn.Module.get_parameter(target)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
module.get_parameter("target")

# Paddle 写法
getattr(module, "target")
```
