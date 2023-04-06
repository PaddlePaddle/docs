## [ 组合替代实现 ]torch.is_nonzero

### [torch.is_nonzero](https://pytorch.org/docs/master/generated/torch.is_nonzero.html#torch.is_nonzero)

```python
torch.is_nonzero(input)
```

用于判断单个元素是否为 `0` 或者 `False` ，当 `input` 的元素个数不为 1 时，抛出 `RuntimeError`; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.is_nonzero(x)

# Paddle 写法
if paddle.numel(x) != 1:
    raise RuntimeError('number of tensor elements must equal to 1!')
else:
    y = paddle.to_tensor(x, dtype='bool').item()
```
