## [组合替代实现]torch.nn.Module.get_submodule

### [torch.nn.Module.get_submodule](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule)

```python
torch.nn.Module.get_submodule(target)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
module.get_submodule("target")

# Paddle 写法
getattr(module, "target")
```
