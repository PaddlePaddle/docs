## [ 用法不同：涉及上下文修改 ]torch.nn.utils.clip_grad_value_
### [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html?highlight=clip_grad_value_#torch.nn.utils.clip_grad_value_)

```python
torch.nn.utils.clip_grad_value_(parameters,
                                clip_value)
```

### [paddle.nn.ClipGradByValue](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ClipGradByValue_cn.html#clipgradbyvalue)

```python
paddle.nn.ClipGradByValue(max,
                          min=None)
```

其中 Pytorch 的 clip_value 与 Paddle 的 max 用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| parameters    | -            | 表示要操作的 Tensor，Pytorch 属于原位操作，PaddlePaddle 无此参数，需要实例化之后在 optimizer 中设置才可以使用。  |
| clip_value    | max            | 表示裁剪梯度的范围，范围为 $[-clip_value, clip_vale]$，PaddlePaddle 的 max 参数可对应实现该参数功能。  |
| -             | min          | 表示裁剪梯度的最小值，PyTorch 无此参数，Paddle 保持默认即可。  |

### 转写示例
```python
# torch 用法
net = Model()
sgd = torch.optim.SGD(net.parameters(), lr=0.1)
for i in range(10):
    loss = net(x)
    loss.backward()
    torch.nn.utils.clip_grad_value_(net.parameters(), 1.)
    sgd.step()

# paddle 用法
net = Model()
sgd = paddle.optim.SGD(net.parameters(), lr=0.1, grad_clip=paddle.nn.ClipGradByValue(), 1.)
for i in range(10):
    loss = net(x)
    loss.backward()
    sgd.step()
```
