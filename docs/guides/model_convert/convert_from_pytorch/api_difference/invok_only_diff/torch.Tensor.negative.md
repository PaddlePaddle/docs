## [ 仅 API 调用方式不一致 ]torch.Tensor.negative

### [torch.Tensor.negative](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.negative)

```python
torch.Tensor.negative()
```

### [paddle.Tensor.neg](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/neg_cn.html#paddle/Tensor/neg_cn#cn-api-paddle-Tensor-neg)

```python
paddle.Tensor.neg(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.negative()

# Paddle 写法
result = a.neg()
```
