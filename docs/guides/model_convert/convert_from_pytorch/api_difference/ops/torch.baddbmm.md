## [ 组合替代实现 ]torch.baddbmm

### [torch.baddbmm](https://pytorch.org/docs/stable/generated/torch.baddbmm.html?highlight=baddbmm#torch.baddbmm)

```python
torch.baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None)
```
Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)

# Paddle 写法
y = beta * input + alpha * paddle.bmm(batch1, batch2)
```
