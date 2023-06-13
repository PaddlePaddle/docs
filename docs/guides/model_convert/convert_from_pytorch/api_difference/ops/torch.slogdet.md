## [ 仅参数名不一致 ]torch.slogdet
### [torch.slogdet](https://pytorch.org/docs/stable/generated/torch.slogdet.html?highlight=slogdet#torch.slogdet)

```python
torch.slogdet(input *, out=None)
```

### [paddle.linalg.slogdet](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/slogdet_cn.html#slogdet)

```python
paddle.linalg.slogdet(x)
```

两者功能一致但参数类型不一致，Pytorch 返回 named tuple，Paddle 返回 Tensor，需要转写。具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | - | 表示输出的 Tuple ，Paddle 无此参数，暂无转写方式。  |


### 转写示例
#### 返回类型不一致
```python
# Pytorch 写法
y = torch.slogdet(a)

# Paddle 写法
result = paddle.linalg.slogdet(a)
y = tuple([result[0], result[1]])
```
