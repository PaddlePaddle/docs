## [torch 参数更多 ]torch.cat
### [torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html?highlight=cat#torch.cat)

```python
torch.cat(input,
           *,
           out=None)
```

### [paddle.concat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/concat_cn.html#concat)

```python
paddle.concat(x,
              name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>| <font color='red'>x</font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'>out</font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.cat([x, y], out=y)

# Paddle 写法
paddle.assign(paddle.concat([x, y]), y)
```
