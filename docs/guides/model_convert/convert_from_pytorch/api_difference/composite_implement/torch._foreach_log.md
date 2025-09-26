## [组合替代实现]torch.\_foreach_log

### [torch.\_foreach_log](https://pytorch.org/docs/stable/generated/torch._foreach_log.html#torch-foreach-log)

```python
torch._foreach_log(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_log(tensors)

# Paddle 写法
[paddle.log(x) for x in tensors]
```
