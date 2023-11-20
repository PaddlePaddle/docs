## [ 组合替代实现 ]torch.nn.init.eye_

### [torch.nn.init.eye_](https://pytorch.org/docs/stable/nn.init.html?highlight=eye_#torch.nn.init.eye_)

```python
torch.nn.init.eye_(tensor)
```

### [paddle.nn.initializer.Assign](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/Assign_cn.html)

```python
paddle.nn.initializer.Assign(value,
                            name=None)
```

两者用法不同：torch 是 inplace 的用法，paddle 是类设置的，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| -          |  value          | 用于初始化参数的一个 Numpy 数组、Python 列表、Tensor。Pytorch 无此参数。此处 Paddle 应使用 paddle.eye 进行参数设置。             |

### 转写示例
```python
# Pytorch 写法
w = torch.empty(3, 5)
torch.nn.init.eye_(w)

# Paddle 写法
w = paddle.empty([3, 5])
init_eye = paddle.nn.initializer.Assign(paddle.eye(w.shape[0], w.shape[1]))
init_eye(w)
```
