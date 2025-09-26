## [ 组合替代实现 ]torch.addr

### [torch.addr](https://pytorch.org/docs/stable/generated/torch.addr.html?highlight=addr#torch.addr)

```python
torch.addr(input, vec1, vec2, beta=1, alpha=1, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.addr(input, vec1, vec2, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.outer(vec1, vec2)
```
