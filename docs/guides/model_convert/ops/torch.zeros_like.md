## torch.zeros_like
### [torch.zeros_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like)

```python
torch.zeros_like(input,
                 *,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False,
                 memory_format=torch.preserve_format)
```

### [paddle.zeros_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/zeros_like_cn.html#zeros-like)

```python
paddle.zeros_like(x,
                  dtype=None,
                  name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。                                     |
| layout        | -            | 表示布局方式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| device        | -            | 表示 Tensor 存放位置，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle 无此参数。 |
| memory_format | -            | 表示内存格式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。           |


### 转写示例
#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.zeros_like((3, 5), requires_grad=True)

# Paddle 写法
x = paddle.zeros_like([3, 5])
x.stop_gradient = False
```
