## [组合替代实现]torch.linalg.tensorinv

### [torch.linalg.inv](https://pytorch.org/docs/stable/generated/torch.linalg.tensorinv.html#torch-linalg-tensorinv)

```python
torch.linalg.tensorinv(A, ind=2, *, out=None)
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

## 转写示例


```python
# PyTorch 写法:
A = torch.eye(4 * 6).reshape((4, 6, 8, 3))
A_inv = torch.linalg.tensorinv(A, ind=2)

# Paddle 转写：
A = paddle.eye(4 * 6).reshape((4, 6, 8, 3))
ind = 2
A_dim_0 = prod(A.shape[ind:]) # 4 * 6 == 24
A_dim_1 = prod(A.shape[:ind]) # 8 * 3 == 24
# assert A_dim_0 == A_dim_1
A_ = paddle.reshape(A, [A_dim_0, A_dim_1])
A_inv_ = paddle.linalg.inv(A_)

A_inv = paddle.reshape(A_inv_, A.shape[ind:] + A.shape[:ind])
```
