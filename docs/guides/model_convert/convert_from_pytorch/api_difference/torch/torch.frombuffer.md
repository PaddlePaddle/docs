## [ 组合替代实现 ]torch.frombuffer

### [torch.frombuffer](https://pytorch.org/docs/stable/generated/torch.frombuffer.html)

```python
torch.frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False)
```

把 Python 缓冲区创建的的对象变成一维 Tensor，Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.frombuffer(a, dtype=torch.int32)

# Paddle 写法
paddle.to_tensor(np.frombuffer(a, dtype='int32'))
```
