## [无参数] torch.Tensor.is_cuda

### [torch.Tensor.is_cuda](https://pytorch.org/docs/stable/generated/torch.Tensor.is_cuda.html?highlight=is_cuda#torch.Tensor.is_cuda)

```
torch.Tensor.is_cuda
```

### [paddle.Tensor.place](no)

```
paddle.Tensor.place
```

两者返回内容不一致，torch 返回的是 bool 型，为了实现此功能，以下为 paddle 版本的替代实现方案

### 转写示例

```
# PyTorch 写法
d = torch.Tensor([1,2,3])
d.is_cuda

# Paddle 写法
d = paddle.to_tensor([1,2,3])
"gpu" in str(d.place)
```
