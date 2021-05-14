# 飞桨框架昇腾NPU版训练示例


## Mnist运行示例：

### 下载模型和运行脚本
```
# 下载模型和数据
wget https://fleet.bj.bcebos.com/ascend/npu.tar.gz
tar xzvf npu.tar.gz
```

### 执行训练
```
sh run_mnist.sh
```

### 配置说明

run_mnist.sh中的配置
```
# 配置paddle的路径
export PYTHONPATH=/your/dir/Paddle/build/python:$PYTHONPATH
# 配置使用的NPU卡
export FLAGS_selected_npus=5
# 执行训练脚本
python3 mnist.py
```
