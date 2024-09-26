## [ 组合替代实现 ]torch.distributions.transforms.PositiveDefiniteTransform

### [torch.distributions.transforms.PositiveDefiniteTransform](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.transforms)

```python
torch.distributions.transforms.PositiveDefiniteTransform(cache_size=0)
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                |
| ---------- | ------------ | ------------------------------------------------------------------- |
| cache_size | -            | 缓存大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

```python
# PyTorch 写法
y = PositiveDefiniteTransform()(tensor1)

# Paddle 写法
T = tensor1.tril(-1) + tensor1.diagonal(-2, -1).exp().diag_embed()
y = T @ T.mT
```
