## [ 仅 API 调用方式不一致 ]torch.Tensor.positive

### [torch.Tensor.positive](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.positive)

```python
torch.Tensor.positive()
```

### [paddle.positive](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/positive_cn.html#paddle/positive_cn#cn-api-paddle-positive)

```python
paddle.positive(x)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.positive()

# Paddle 写法
result = paddle.positive(x)
```
