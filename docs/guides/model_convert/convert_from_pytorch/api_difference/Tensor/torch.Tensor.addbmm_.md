## [ 组合替代实现 ]torch.Tensor.addbmm_

### [torch.Tensor.addbmm_](https://pytorch.org/docs/stable/generated/torch.Tensor.addbmm_.html#torch.Tensor.addbmm_)

```python
torch.Tensor.addbmm_(batch1, batch2, *, beta=1, alpha=1)
```

用于实现矩阵 `batch1` 与矩阵 `batch2` 相乘，将结果按 `axis=0` 求和之后与`alpha`相乘，再加上输入 `input` 与 `beta`，公式为：

$$
out = \beta \, input + \alpha \left( \sum_{i=0}^{b-1} batch1_i \, @ \, batch2_i \right)
$$

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
input.addbmm_(batch1, batch2, beta=beta, alpha=alpha)

# Paddle 写法
input.multiply_(paddle.to_tensor(beta, dtype=input.dtype)).add_(alpha * paddle.sum(paddle.bmm(batch1, batch2), axis=0))
```
