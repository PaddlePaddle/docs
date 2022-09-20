# 飞桨框架昆仑 XPU 版训练示例

使用 XPU 训练与 cpu/gpu 相同，只需要加上-o use_xpu=True, 表示执行在昆仑设备上。

#### ResNet50 下载并运行示例：

模型文件下载命令：

```
cd path_to_clone_PaddleClas
git clone -b release/static https://github.com/PaddlePaddle/PaddleClas.git
```
也可以访问 PaddleClas 的[github repo](https://github.com/PaddlePaddle/PaddleClas/tree/release/static)直接下载源码。

配置 XPU 进行训练的命令非常简单：
```
#FLAGS 指定单卡或多卡训练，此示例运行 2 个卡
export FLAGS_selected_xpus=0,1
#启动训练
python3.7 tools/static/train.py -c configs/quick_start/ResNet50_vd_finetune_kunlun.yaml -o use_gpu=False -o use_xpu=True -o is_distributed=False
```

如果需要指定更多的卡（比如 8 卡），需要配置合适的训练参数，可使用如下命令自动修改：
```
export FLAGS_selected_xpus=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch \
        --ips=${ips} \
        --xpus=${FLAGS_selected_xpus} \
        --log_dir log \
        tools/static/train.py \
        -c ${config_yaml} \
        -o is_distributed=False \
        -o epochs=${epochs} \
        -o TRAIN.batch_size=${total_batch_size} \
        -o LEARNING_RATE.params.lr=${lr} \
        -o use_gpu=False \
        -o use_xpu=True
```

其他模型的训练示例可在[飞桨对昆仑 XPU 芯片的支持](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_2.0_xpu_cn.html)中支持模型列表下的模型链接中找到。
