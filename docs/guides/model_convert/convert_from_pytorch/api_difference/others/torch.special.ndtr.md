## [ 组合替代实现 ]torch.special.ndtr

### [torch.special.ndtr](https://pytorch.org/docs/stable/special.html#torch.special.ndtr)

```python
torch.special.ndtr(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

对应公式：
$$
\operatorname{ndtr}(x)=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} e^{-\frac{1}{2} t^{2}} d t
$$

### 转写示例

```python
# PyTorch 写法
torch.special.ndtr(a)

# Paddle 写法
(paddle.erf(a/paddle.sqrt(paddle.to_tensor(2)))-paddle.erf(paddle.to_tensor(-float('inf'))))/2
```
