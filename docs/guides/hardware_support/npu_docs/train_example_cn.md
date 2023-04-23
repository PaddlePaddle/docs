# 飞桨框架昇腾 NPU 版训练示例

## YOLOv3 训练示例

**第一步**：下载并安装 PaddleDetection 套件

```bash
# 下载套件代码
cd path_to_clone_PaddleDetection
git clone -b develop https://github.com/PaddlePaddle/PaddleDetection.git

# 编译安装
cd PaddleDetection
python setup.py install

# 安装其他依赖
pip install -r requirements.txt
```

也可以访问 PaddleDetection 的 [GitHub Repo](https://github.com/PaddlePaddle/PaddleDetection) 下载 develop 分支的源码。

**第二步**：准备 VOC 训练数据集

```bash
cd PaddleDetection/static/dataset/roadsign_voc
python download_roadsign_voc.py

# 下载完成之后，当前目录结构如下
PaddleDetection/static/dataset/roadsign_voc/
├── annotations
├── download_roadsign_voc.py
├── images
├── label_list.txt
├── train.txt
└── valid.txt
```

**第三步**：运行单卡训练

```bash
export FLAGS_selected_npus=0

# 单卡训练
python -u tools/train.py -c configs/yolov3_darknet_roadsign.yml -o use_npu=True

# 单卡评估
python -u tools/eval.py -c configs/yolov3_darknet_roadsign.yml -o use_npu=True

# 精度结果
INFO:ppdet.utils.voc_eval:mAP(0.50, integral) = 76.78%
```

**第四步**：运行多卡训练

> 注意：多卡训练请参考本页下一章节进行 "NPU 多卡训练配置" 的准备。

```bash
# NPU 多卡训练配置
export FLAGS_selected_npus=0,1,2,3
export RANK_TABLE_FILE=/root/hccl_4p_0123_127.0.0.1.json

# 设置 HCCL 相关环境变量
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export HCCL_SECURITY_MODE=1

# 多卡训练
python -m paddle.distributed.fleet.launch --run_mode=collective \
       tools/train.py -c configs/yolov3_darknet_roadsign.yml -o use_npu=True

# 多卡训练结果评估
python -u tools/eval.py -c configs/yolov3_darknet_roadsign.yml -o use_npu=True

# 精度结果
INFO:ppdet.utils.voc_eval:mAP(0.50, integral) = 83.00%
```

## NPU 多卡训练配置

**预先要求**：请先根据华为昇腾 910 NPU 的文档 [配置 device 的网卡 IP](https://support.huaweicloud.com/instg-cli-cann502-alpha005/atlasdeploy_03_0105.html) 进行相关 NPU 运行环境的部署和配置，配置完成后检查机器下存在 `/etc/hccn.conf` 文件。

如果是物理机环境，请根据华为官网的 [hccl_tools 说明文档](https://github.com/mindspore-ai/mindspore/tree/v1.4.0/model_zoo/utils/hccl_tools) 进行操作。如果是根据 Paddle 官方镜像启动的容器环境，请根据以下步骤进行操作：

**第一步**：根据容器启动时映射的设备 ID，创建容器内的 `/etc/hccn.conf` 文件

例如物理机上的 8 卡的原始 `/etc/hccn.conf` 文件内容如下：

```
address_0=192.168.10.21
netmask_0=255.255.255.0
address_1=192.168.20.21
netmask_1=255.255.255.0
address_2=192.168.30.21
netmask_2=255.255.255.0
address_3=192.168.40.21
netmask_3=255.255.255.0
address_4=192.168.10.22
netmask_4=255.255.255.0
address_5=192.168.20.22
netmask_5=255.255.255.0
address_6=192.168.30.22
netmask_6=255.255.255.0
address_7=192.168.40.22
netmask_7=255.255.255.0
```

容器启动命令中映射的设备 ID 为 4 到 7 的 4 张 NPU 卡，则创建创建容器内的 `/etc/hccn.conf` 文件内容如下：

> 注意：这里的 address_4 和 netmask_4 需要相应的修改为 address_0 和 netmask_0，以此类推

```
address_0=192.168.10.22
netmask_0=255.255.255.0
address_1=192.168.20.22
netmask_1=255.255.255.0
address_2=192.168.30.22
netmask_2=255.255.255.0
address_3=192.168.40.22
netmask_3=255.255.255.0
```

**第二步**：根据华为官网的 [hccl_tools 说明文档](https://github.com/mindspore-ai/mindspore/tree/v1.4.0/model_zoo/utils/hccl_tools)，生成单机四卡的配置文件

```
# 下载 hccl_tools.py 文件到本地
wget https://raw.githubusercontent.com/mindspore-ai/mindspore/v1.4.0/model_zoo/utils/hccl_tools/hccl_tools.py

# 生成单机两卡的配置文件，单机可以设置 IP 为 127.0.0.1
python hccl_tools.py --device_num "[0,4)" --server_ip 127.0.0.1
```

运行成功之后在当前目录下获得名为 `hccl_4p_0123_127.0.0.1.json` 的文件，内容如下：

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "127.0.0.1",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.168.10.22",
                    "rank_id": "0"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.20.22",
                    "rank_id": "1"
                },
                {
                    "device_id": "2",
                    "device_ip": "192.168.30.22",
                    "rank_id": "2"
                },
                {
                    "device_id": "3",
                    "device_ip": "192.168.40.22",
                    "rank_id": "3"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

**第三步**：运行 Paddle 多卡训练之前，需要先配置名为 `RANK_TABLE_FILE` 的环境变量，指向上一步生成的 json 文件的绝对路径

```bash
# 1) 设置 ranktable 文件的环境变量
export RANK_TABLE_FILE=$(readlink -f hccl_4p_0123_127.0.0.1.json)
# 或者直接修改为 json 文件的绝对路径
export RANK_TABLE_FILE=/root/hccl_4p_0123_127.0.0.1.json

# 2) 设置 HCCL 相关环境变量
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export HCCL_SECURITY_MODE=1

# 3) 启动分布式任务，注意这里的 run_mode 当前仅支持 collective 模式
python -m paddle.distributed.fleet.launch --run_mode=collective train.py ...
```
