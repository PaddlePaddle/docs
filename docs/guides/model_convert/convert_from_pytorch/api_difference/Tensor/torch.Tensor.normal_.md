## [组合替代实现]torch.Tensor.normal_

### [torch.Tensor.norm_](https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html#torch-tensor-normal)

```python
torch.Tensor.normal_(mean=0, std=1, *, generator=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
x.normal_(mean=0)

# Paddle 写法
paddle.assign(paddle.normal(mean=0, shape=x.shape).astype(x.dtype), x)
```
