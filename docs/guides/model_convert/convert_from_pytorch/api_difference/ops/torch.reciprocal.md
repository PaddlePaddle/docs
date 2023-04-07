## [torch 参数更多]torch.reciprocal
### [torch.reciprocal](https://pytorch.org/docs/stable/generated/torch.reciprocal.html?highlight=reciprocal#torch.reciprocal)

```python
torch.reciprocal(input,
                 *,
                 out=None)
```

### [paddle.reciprocal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/reciprocal_cn.html#reciprocal)

```python
paddle.reciprocal(x,
                  name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.reciprocal([3, 5], out=y)

# Paddle 写法
paddle.assign(paddle.reciprocal([3, 5]), y)
```
