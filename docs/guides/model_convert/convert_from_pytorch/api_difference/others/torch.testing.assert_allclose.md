## [ 组合替代实现 ]torch.testing.assert_allclose

### [torch.testing.assert_allclose](https://pytorch.org/docs/stable/testing.html?highlight=torch+testing+assert_allclose#torch.testing.assert_allclose)

```python
torch.testing.assert_allclose(actual, expected, rtol=None, atol=None, equal_nan=True, msg='')
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
torch.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True, msg='error messege')

# Paddle 写法
assert paddle.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True).item(), 'error messege'
```
