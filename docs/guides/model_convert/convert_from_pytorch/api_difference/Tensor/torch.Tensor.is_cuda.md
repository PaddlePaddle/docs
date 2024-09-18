## [组合替代实现] torch.Tensor.is_cuda

### [torch.Tensor.is_cuda](https://pytorch.org/docs/stable/generated/torch.Tensor.is_cuda.html?highlight=is_cuda#torch.Tensor.is_cuda)

```python
torch.Tensor.is_cuda
```

判断 Tensor 是否在 gpu 上，PaddlePaddle 目前无对应 API，可使用如下代码组合替代实现:

### 转写示例

```python
# PyTorch 写法
d = torch.Tensor([1,2,3])
d.is_cuda

# Paddle 写法
d = paddle.to_tensor([1,2,3])
d.place.is_gpu_place()
```
