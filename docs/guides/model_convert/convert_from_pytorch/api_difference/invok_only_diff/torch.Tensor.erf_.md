## [ 仅 API 调用方式不一致 ]torch.Tensor.erf_

### [torch.Tensor.erf\_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.erf_.html#torch.Tensor.erf_)

```python
torch.Tensor.erf_()
```

### [paddle.erf\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/erf__cn.html#paddle.erf_)

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
