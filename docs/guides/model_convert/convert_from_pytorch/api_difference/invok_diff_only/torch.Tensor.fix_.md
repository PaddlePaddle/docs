## [ 仅 API 调用方式不一致 ]torch.Tensor.fix_

### [torch.Tensor.fix_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.fix_)

```python
torch.Tensor.fix_()
```

### [paddle.Tensor.trunc_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/trunc__cn.html#paddle/Tensor/trunc__cn#cn-api-paddle-Tensor-trunc_)

```python
paddle.Tensor.trunc_(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = input.fix_()

# Paddle 写法
result = input.trunc_()
```
