## [ 组合替代实现 ]torch.distributions.transforms.PositiveDefiniteTransform

### [torch.distributions.transforms.PositiveDefiniteTransform](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.transforms)

```python
torch.distributions.transforms.PositiveDefiniteTransform(cache_size=0)
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 转写示例

```python
# PyTorch 写法
y = PositiveDefiniteTransform()(tensor1)

# Paddle 写法
T = tensor1.tril(-1) + tensor1.diagonal(-2, -1).exp().diag_embed()
y = T @ T.mT
```
