## [ 组合替代实现 ]torch.Size.numel

### [torch.Size.numel](https://pytorch.org/docs/stable/size.html)

```python
torch.Size.numel()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
x = torch.ones(10, 20, 30)
s = x.size()
s.numel()

# Paddle 写法
x = paddle.ones([10, 20, 30])
result = tuple(x.shape)
out = 1
for x in result:
    out *= x
```
