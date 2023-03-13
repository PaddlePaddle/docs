## torch.pow
### [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html?highlight=pow#torch.pow)

```python
torch.pow(input,
          exponent,
          *,
          out=None)
```

### [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/pow_cn.html#pow)

```python
paddle.pow(x,
           y,
           name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| exponent      | y            | 指数值。                                             |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.pow([3, 5], 2, out=y)

# Paddle 写法
y = paddle.pow([3, 5], 2)
```
