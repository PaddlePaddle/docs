## [ 仅 API 调用方式不一致 ]torch.Tensor.expm1_

### [torch.Tensor.expm1_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expm1_)

```python
torch.Tensor.expm1_()
```

### [paddle.expm1_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/expm1__cn.html#paddle/expm1__cn#cn-api-paddle-expm1_)

```python
paddle.expm1_(x, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
a.expm1_()

# Paddle 写法
paddle.expm1_(a)
```
