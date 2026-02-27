## [ 组合替代实现 ]torch.testing.assert_close
### [torch.testing.assert\_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)
```python
torch.testing.assert_close(actual, expected, *, allow_subclasses=True, rtol=None, atol=None, equal_nan=False, check_device=True, check_dtype=True, check_layout=True, check_stride=False, msg=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# PyTorch 写法
torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True, msg='error message')

# Paddle 写法
assert paddle.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True).item(), 'error message'
```
