## [组合替代实现]torch.positive

### [torch.positive](https://pytorch.org/docs/stable/generated/torch.positive.html#torch-positive)

```python
torch.positive(input)
```

判断 `input` 是否是 bool 类型的 Tensor，如果是则抛出 RuntimeError 异常，否则返回 `input` 。

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.positive(x)

# Paddle 写法
def positive(x):
    if x.dtype != paddle.bool:
        return x
    else:
        raise RuntimeError("boolean tensors is not supported.")

positive(x)
```
