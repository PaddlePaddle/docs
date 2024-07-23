## [ 组合替代实现 ]torch.is_nonzero

### [torch.is_nonzero](https://pytorch.org/docs/stable/generated/torch.is_nonzero.html#torch.is_nonzero)

```python
torch.is_nonzero(input)
```

用于判断单个元素是否为 `0` 或者 `False` ，当 `input` 的元素个数不为 1 时，抛出 `RuntimeError`; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.is_nonzero(x)

# Paddle 写法
x.astype('bool').item()
```
