## [仅 torch 参数更多]torch.addmm

### [torch.addmm](https://pytorch.org/docs/stable/generated/torch.addmm.html?highlight=addmm#torch.addmm)

```python
torch.addmm(input,mat1,mat2,*,beta=1,alpha=1,out=None)
```

### [paddle.addmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/addmm_cn.html)

```python
paddle.addmm(input,x,y,alpha=1.0,beta=1.0,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|input|input|表示输入的 Tensor 。|
|mat1|x|表示输入的第一个 Tensor ，仅参数名不一致。|
|mat2|y|表示输入的第二个 Tensor ，仅参数名不一致。|
|beta|beta|表示乘以 input 的标量。|
|alpha|alpha|表示乘以 mat1 * mat2 的标量。|
|out||表示输出的 Tensor ， Paddle 无此参数，需要进行转写。|

### 转写示例

#### out: 输出的 Tensor

```python
# Pytorch 写法
torch.addmm(input,x,y,beta,alpha,out=output)

# Paddle 写法
paddle.assign(paddle.addmm(input,x,y,beta,alpha),output)
```
