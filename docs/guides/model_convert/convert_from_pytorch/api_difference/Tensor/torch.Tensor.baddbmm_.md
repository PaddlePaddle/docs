## [ 组合替代实现 ]torch.Tensor.baddbmm_

### [torch.Tensor.baddbmm_](https://pytorch.org/docs/stable/generated/torch.Tensor.baddbmm_.html#torch.Tensor.baddbmm_)

```python
torch.Tensor.baddbmm_(batch1, batch2, beta=1, alpha=1)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
input.baddbmm_(batch1, batch2, beta=beta, alpha=alpha)

# Paddle 写法
paddle.assign(beta * input + alpha * paddle.bmm(batch1, batch2), input)
```
