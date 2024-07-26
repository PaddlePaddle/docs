## [ 输入参数类型不一致 ]torch.Tensor.remainder_

### [torch.Tensor.remainder_](https://pytorch.org/docs/stable/generated/torch.Tensor.remainder_.html?highlight=torch+tensor+remainder_#torch.Tensor.remainder_)

```python
torch.Tensor.remainder_(other)
```

### [paddle.Tensor.remainder_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id15)

```python
paddle.Tensor.remainder_(y, name=None)
```

其中 Paddle 与 PyTorch 运算除数参数所支持的类型不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| other         | y            | 除数，PyTorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor，需要转写。   |

### 转写示例

#### other：除数为 Scalar

```python
# PyTorch 写法
x.remainder_(1)

# Paddle 写法
x.remainder_(y=paddle.to_tensor(1))
```
