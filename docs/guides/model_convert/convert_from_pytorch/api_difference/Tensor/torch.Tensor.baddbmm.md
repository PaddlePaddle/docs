## [ 组合替代实现 ]torch.Tensor.baddbmm

### [torch.Tensor.baddbmm](https://pytorch.org/docs/stable/generated/torch.Tensor.baddbmm.html#torch.Tensor.baddbmm)

```python
torch.Tensor.baddbmm(batch1, batch2, beta=1, alpha=1)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = input.baddbmm(batch1, batch2, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.bmm(batch1, batch2)
```
