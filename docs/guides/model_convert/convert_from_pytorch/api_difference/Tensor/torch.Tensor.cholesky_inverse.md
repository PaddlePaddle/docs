## [ 组合替代实现 ]torch.cholesky_inverse

### [torch.Tensor.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch.cholesky_inverse)
```python
torch.Tensor.cholesky_inverse(upper=False)
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 转写示例
```python
# Pytorch 写法
y = u.cholesky(upper)

# Python 写法
ut = paddle.transpose(u, perm=[1, 0])
if upper:
    out = paddle.linalg.inv(paddle.matmul(ut, u))
else:
    out = paddle.inverse(paddle.matmul(u, ut))
```
