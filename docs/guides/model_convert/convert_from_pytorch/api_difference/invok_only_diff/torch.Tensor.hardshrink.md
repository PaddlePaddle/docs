## [ 仅 API 调用方式不一致 ]torch.Tensor.hardshrink

### [torch.Tensor.hardshrink](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.hardshrink)

```python
torch.Tensor.hardshrink()
```

### [paddle.nn.functional.hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hardshrink_cn.html#paddle/nn/functional/hardshrink_cn#cn-api-paddle-nn-functional-hardshrink)

```python
paddle.nn.functional.hardshrink(x, threshold=0.5, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.hardshrink()

# Paddle 写法
result = paddle.nn.functional.hardshrink(x)
```
