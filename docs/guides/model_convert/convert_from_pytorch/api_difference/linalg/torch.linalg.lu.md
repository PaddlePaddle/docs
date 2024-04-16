## [ 组合替代实现 ]torch.linalg.lu

### [torch.linalg.lu](https://pytorch.org/docs/stable/generated/torch.linalg.lu.html?highlight=torch+linalg+lu#torch.linalg.lu)

```python
torch.linalg.lu(A, *, pivot=True, out=None)
```

Paddle 无此 API，需要组合实现。
PyTorch 中 torch.linalg.lu 返回值为 (P, L, U)，Paddle 中 paddle.linalg.lu 返回值为(LU, P)，需要转写。

### 转写示例

```python
# PyTorch 写法
P, L, U = torch.linalg.lu(x)

# Paddle 写法
lu, p = paddle.linalg.lu(x)
P, L, U = paddle.linalg.lu_unpack(lu, p)
```
