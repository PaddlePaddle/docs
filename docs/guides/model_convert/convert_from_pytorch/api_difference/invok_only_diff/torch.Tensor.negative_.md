## [ 仅 API 调用方式不一致 ]torch.Tensor.negative_

### [torch.Tensor.negative\_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.negative_.html#torch.Tensor.negative_)

```python
torch.Tensor.negative_()
```

### [paddle.Tensor.neg_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/neg__cn.html#paddle/Tensor/neg__cn#cn-api-paddle-Tensor-neg_)

```python
paddle.Tensor.neg_(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
a.negative_()

# Paddle 写法
a.neg_()
```
