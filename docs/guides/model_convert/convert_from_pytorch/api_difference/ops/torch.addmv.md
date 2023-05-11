## [ 组合替代实现 ]torch.addmv

### [torch.addmv](https://pytorch.org/docs/stable/generated/torch.addmv.html?highlight=addmv#torch.addmv)
```python
torch.addmv(input, mat, vec, beta=1, alpha=1, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.addmv(input, mat, vec, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.mm(mat, vec)
```
