## torch.bernoulli
### [torch.bernoulli](https://pytorch.org/docs/stable/generated/torch.bernoulli.html?highlight=bernoulli#torch.bernoulli)

```python
torch.bernoulli(input,
                *,
                generator=None,
                out=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bernoulli_cn.html#bernoulli)

```python
paddle.bernoulli(x,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。                                     |
| generator     | -            | 用于采样的伪随机数生成器，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。          |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.bernoulli([3, 5], out=y)

# Paddle 写法
y = paddle.bernoulli([3, 5])
```
