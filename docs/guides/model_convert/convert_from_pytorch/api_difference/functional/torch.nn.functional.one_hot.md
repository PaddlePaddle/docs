## [ 仅参数默认值不一致 ]torch.nn.functional.one_hot

### [torch.nn.functional.one_hot](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html?highlight=one_hot#torch.nn.functional.one_hot)

```python
torch.nn.functional.one_hot(tensor,
                            num_classes=- 1)
```

### [paddle.nn.functional.one_hot](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/one_hot_cn.html)

```python
paddle.nn.functional.one_hot(x,
                             num_classes,
                             name=None)
```

两者功能一致，但 Paddle 的 num_classes 没有指定默认值，需要手动指定，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor          | x         | 表示输入的 Tensor 。                                     |
| num_classes | num_classes | 表示一个 one-hot 向量的长度， Paddle 为非默认参数，需要指定为与 Pytorch 一致。 |

### 转写示例
```python
# PyTorch 写法
y = torch.nn.functional.one_hot(x)

# Paddle 写法
y = paddle.nn.functional.one_hot(x, num_classes=-1)
```
