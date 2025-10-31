## [ 仅 API 调用方式不一致 ]torch.Tensor.retain_grad

### [torch.Tensor.retain_grad](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.retain_grad)

```python
torch.Tensor.retain_grad()
```

### [paddle.Tensor.retain_grads](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/retain_grads_cn.html#paddle/Tensor/retain_grads_cn#cn-api-paddle-Tensor-retain_grads)

```python
paddle.Tensor.retain_grads()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result.retain_grad()

# Paddle 写法
result.retain_grads()
```
