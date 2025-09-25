## [组合替代实现]torch.utils.set_module

### [torch.utils.set_module](https://docs.pytorch.org/docs/stable/generated/torch.utils.set_module.html#torch-utils-set-module)

```python
torch.utils.set_module(obj, mod)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.utils.set_module(obj, mod)

# Paddle 写法
obj.__module__ = mod
```
