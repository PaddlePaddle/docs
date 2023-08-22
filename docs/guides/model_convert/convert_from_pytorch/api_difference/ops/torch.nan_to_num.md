## [torch 参数更多 ]torch.nan_to_num

### [torch.nan_to_num](https://pytorch.org/docs/stable/generated/torch.nan_to_num.html?highlight=nan_to_num#torch.nan_to_num)

```python
torch.nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None)
```

### [paddle.nan_to_num](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nan_to_num_cn.html#nan-to-num)

```python
paddle.nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input  | x  | 表示输入的 Tensor ，仅参数名不一致。  |
| nan  |  nan  | 表示用于替换 nan 的值。  |
| posinf  |  posinf  | 表示+inf 的替换值。  |
| neginf  |  neginf  | 表示-inf 的替换值。  |
| out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.nan_to_num(x, n, pos, neg, out = y)

# Paddle 写法
paddle.assign(paddle.nan_to_num(x, n, pos, neg), y)
```
