## torch.multinomial
### [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial#torch.multinomial)
```python
torch.multinomial(input,
                  num_samples,
                  replacement=False,
                  *,
                  generator=None,
                  out=None)
```
### [paddle.multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multinomial_cn.html#multinomial)
```python
paddle.multinomial(x,
                   num_samples=1,
                   replacement=False,
                   name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。                                     |
| generator     | -            | 用于采样的伪随机数生成器，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.multinomial([0.3, 0.5, 0.2], out=y)

# Paddle 写法
y = paddle.multinomial([0.3, 0.5, 0.2])
```
