## [ 仅 API 调用方式不一致 ]torch.Tensor.retain_grad

### [torch.Tensor.retain\_grad](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html#torch.Tensor.retain_grad)

```python
torch.Tensor.retain_grad()
```

### [paddle.Tensor.retain\_grads](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#retain-grads)

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
