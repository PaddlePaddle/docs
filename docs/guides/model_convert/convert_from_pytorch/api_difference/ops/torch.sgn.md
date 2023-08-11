## [torch 参数更多 ]torch.sgn
### [torch.sgn](https://pytorch.org/docs/stable/generated/torch.sgn.html?highlight=torch+sgn#torch.sgn)

```python
torch.sgn(input,
          *,
          out=None)
```

### [paddle.sgn](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sgn_cn.html)

```python
paddle.sgn(x,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.sgn([3, 5], out=y)

# Paddle 写法
paddle.assign(paddle.sgn([3, 5]), y)
```
