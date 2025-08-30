## [组合替代实现]torch.Tensor.requires_grad

### [torch.Tensor.requires_grad](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html#torch-tensor-requires-grad)

```python
torch.Tensor.requires_grad
```

### [paddle.Tensor.stop_gradient](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#stop-gradient)

```python
paddle.Tensor.stop_gradient
```

Paddle 无此 API，需要组合实现。

### 转写示例
```python
# 当 torch 写法
x.requires_grad = True

# paddle 写法
x.stop_gradient = False

# 当 torch 写法
x.requires_grad = False

# paddle 写法
x.stop_gradient = True

# torch 写法
x.requires_grad

# paddle 写法
not x.stop_gradient
```
