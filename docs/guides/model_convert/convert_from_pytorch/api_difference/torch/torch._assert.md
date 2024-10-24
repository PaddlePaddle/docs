## [ 组合替代实现 ]torch._assert

### [torch._assert](https://pytorch.org/docs/stable/generated/torch._assert.html)

```python
torch._assert(condition, message)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._assert(condition=1==2, message='error messege')

# Paddle 写法
assert 1==2, 'error messege'
```
