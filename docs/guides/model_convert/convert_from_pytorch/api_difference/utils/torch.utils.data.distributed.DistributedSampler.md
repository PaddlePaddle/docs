## [ 参数不一致 ]torch.utils.data.distributed.DistributedSampler
### [torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler)

```python
torch.utils.data.distributed.DistributedSampler(dataset,
                                                num_replicas=None,
                                                rank=None,
                                                shuffle=True,
                                                seed=0,
                                                drop_last=False)
```

### [paddle.io.DistributedBatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/DistributedBatchSampler_cn.html#distributedbatchsampler)

```python
paddle.io.DistributedBatchSampler(dataset=None,
                                  batch_size,
                                  num_replicas=None,
                                  rank=None,
                                  shuffle=False,
                                  drop_last=False)
```

两者功能一致但参数不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ----- | ---------- | ---------- |
| seed          | -            | 如果 shuffle=True，则使用随机种子对采样器进行随机排序,此数字在分布式组中的所有进程中应相同，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| -             | batch_size   | 每 mini-batch 中包含的样本数，PyTorch 无此参数，Paddle 需设置为 1。                   |
