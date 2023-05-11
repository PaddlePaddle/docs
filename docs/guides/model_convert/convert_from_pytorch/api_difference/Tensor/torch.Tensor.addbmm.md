## [ 组合替代实现 ]torch.Tensor.addbmm

### [torch.Tensor.addbmm](https://pytorch.org/docs/stable/generated/torch.Tensor.addbmm.html#torch.Tensor.addbmm)

```python
torch.Tensor.addbmm(batch1, batch2, *, beta=1, alpha=1)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = input.addbmm(batch1, batch2, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.sum(paddle.bmm(batch1, batch2), axis=0)
```
