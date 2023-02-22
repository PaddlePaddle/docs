## torch.full_like
### [torch.full_like](https://pytorch.org/docs/stable/generated/torch.full_like.html?highlight=full_like#torch.full_like)

```python
torch.full_like(input,
                fill_value,
                *,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False,
                memory_format=torch.preserve_format)
```

### [paddle.full_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_like_cn.html#full-like)

```python
paddle.full_like(x,
                 fill_value,
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
| requires_grad | -            | 表示是否阻断梯度传导，PaddlePaddle 无此参数。 |
| memory_format | -            | 表示是内存格式，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。           |


### 转写示例
#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.full_like([3, 5], 1., requires_grad=True)

# Paddle 写法
x = paddle.full_like([3, 5], 1.)
x.stop_gradient = False
```
