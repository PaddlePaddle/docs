## [ 仅 API 调用方式不一致 ]torch.Tensor.erf_

### [torch.Tensor.erf_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.erf_)

```python
torch.Tensor.erf_()
```

### [paddle.erf_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/erf__cn.html#paddle/erf__cn#cn-api-paddle-erf_)

```python
paddle.erf_(x, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
a.erf_()

# Paddle 写法
paddle.erf_(a)
```
