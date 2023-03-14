## [torch 参数更多 ]torch.round
### [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round)

```python
torch.round(input,
            *,
            decimals=0,
            out=None)
```

### [paddle.round](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/round_cn.html#round)

```python
paddle.round(x,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| decimals      | -            | 要舍入到的小数位数，PaddlePaddle 无此参数，需要进行转写。               |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。               |


### 转写示例
#### decimals：要舍入到的小数位数
```python
# Pytorch 写法
torch.round([3.345, 5.774], decimals=2)

# Paddle 写法
paddle.round(1e2 * [3.345, 5.774]) / 1e2

# 注：Paddle 可使用 10 的 decimals 次方来实现
```

#### out：指定输出
```python
# Pytorch 写法
torch.round([3, 5], out=y)

# Paddle 写法
y = paddle.round([3, 5])
```
