## [ torch 参数更多 ]torch.nn.SyncBatchNorm.convert_sync_batchnorm
### [torch.nn.SyncBatchNorm.convert_sync_batchnorm](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html?highlight=convert_sync_batchnorm#torch.nn.SyncBatchNorm.convert_sync_batchnorm)

```python
torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group=None)
```

### [paddle.nn.SyncBatchNorm.convert_sync_batchnorm(layer)](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SyncBatchNorm_cn.html#convert-sync-batchnorm-layer)

```python
paddle.nn.SyncBatchNorm.convert_sync_batchnorm(layer)
```

Pytorch 参数更多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module           | layer      | 需要转换的模型层， 仅参数名不一致。                                    |
| process_group | -            | 统计信息的同步分别在每个进程组内发生， PaddlePaddle 无此参数，暂无转写方式。         |
