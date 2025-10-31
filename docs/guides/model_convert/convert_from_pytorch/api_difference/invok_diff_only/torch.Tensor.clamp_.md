## [ 仅 API 调用方式不一致 ]torch.Tensor.clamp_

### [torch.Tensor.clamp_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.clamp_)

```python
torch.Tensor.clamp_(min=None, max=None)
```

### [paddle.Tensor.clip_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/clip__cn.html#paddle/Tensor/clip__cn#cn-api-paddle-Tensor-clip_)

```python
paddle.Tensor.clip_(min=None, max=None, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.clamp_(-0.5, 0.5)

# Paddle 写法
result = a.clip_(-0.5, 0.5)
```
