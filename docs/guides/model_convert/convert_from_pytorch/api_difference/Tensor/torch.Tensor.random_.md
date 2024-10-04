## [ 组合替代实现 ] torch.Tensor.random_

### [torch.Tensor.random_](https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html)

```python
torch.Tensor.random_(from=0, to=None, *, generator=None)
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 转写示例

```python
# PyTorch 写法
x.random_(from=0, to=10)

# Paddle 写法
paddle.assign(paddle.cast(paddle.randint(low=0, high=2, shape=x.shape), dtype='float32'), x)
```
