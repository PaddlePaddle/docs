## [ 组合替代实现 ]torch.Size.index

### [torch.Size.index](https://pytorch.org/docs/stable/size.html)

```python
torch.Size.index(value, start=0)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
x = torch.ones(10, 20, 30)
s = x.size()
s.index(30, 1)

# Paddle 写法
x = paddle.ones([10, 20, 30])
s = tuple(x.shape)
s.index(30, 1)
```
