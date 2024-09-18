## [ 组合替代实现 ]torch.Tensor.addbmm_

### [torch.Tensor.addbmm_](https://pytorch.org/docs/stable/generated/torch.Tensor.addbmm_.html#torch.Tensor.addbmm_)

```python
torch.Tensor.addbmm_(batch1, batch2, *, beta=1, alpha=1)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
input.addbmm_(batch1, batch2, beta=beta, alpha=alpha)

# Paddle 写法
paddle.assign(beta * input + alpha * paddle.sum(paddle.bmm(batch1, batch2), axis=0), input)
```
