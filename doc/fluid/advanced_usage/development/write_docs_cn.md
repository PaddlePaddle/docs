# 如何贡献文档

PaddlePaddle非常欢迎您贡献文档。如果您撰写/翻译的文档满足我们的要求，您的文档将会呈现在paddlapaddle.org网站和Github上供PaddlePaddle的用户阅读。

Paddle的文档主要分为以下几个模块：

- 新手入门：包括安装说明、深度学习基础知识、学习资料等，旨在帮助用户快速安装和入门；

- 使用指南：包括数据准备、网络配置、训练、Debug、预测部署和模型库文档，旨在为用户提供PaddlePaddle基本用法讲解；

- 进阶使用：包括服务器端和移动端部署、如何贡献代码/文档、如何性能调优等，旨在满足开发者的需求；

我们的文档支持[reStructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)和[Markdown](https://guides.github.com/features/mastering-markdown/) (GitHub风格)格式的内容贡献。

撰写文档完成后，您可以使用预览工具查看文档在官网显示的效果，以验证您的文档是否能够在官网正确显示。


## 如何使用预览工具

如果您正在修改代码文档（即API），并在Docker容器中使用PaddlePaddle，请在您相应的docker容器中执行下列步骤。因为API的文档生成器依赖于PaddlePaddle。

如果您只改进了文本/媒体内容(不需要安装或构建PaddlePaddle)，或者正在主机上构建PaddlePaddle，请继续在主机上执行下列步骤。

### 1. Clone你希望更新或测试的相关仓库：

首先下载完整的文档存储仓库，其中`--recurse-submodules`会同步更新FluidDoc中的submodule（所有的submodule均在`FluidDoc/external`中），以保证所有文档可以正常显示：

```
git clone --recurse-submodules https://github.com/PaddlePaddle/FluidDoc
```

其他可拉取的存储库有：


```
git clone https://github.com/PaddlePaddle/book.git
git clone https://github.com/PaddlePaddle/models.git
git clone https://github.com/PaddlePaddle/Mobile.git

```

您可以将这些本地副本放在电脑的任意目录下，稍后我们会在启动 PaddlePaddle.org时指定这些仓库的位置。

### 2. 在新目录下拉取 PaddlePaddle.org 并安装其依赖项

在此之前，请确认您的操作系统安装了python的依赖项

以ubuntu系统为例，运行：

```
sudo apt-get update && apt-get install -y python-dev build-essential
```

然后：

```
git clone https://github.com/PaddlePaddle/PaddlePaddle.org.git
cd PaddlePaddle.org/portal
# To install in a virtual environment.
# virtualenv venv; source venv/bin/activate
pip install -r requirements.txt
```

**可选项**：如果你希望实现中英网站转换，以改善PaddlePaddle.org，请安装[GNU gettext](https://www.gnu.org/software/gettext/)

### 3. 在本地运行 PaddlePaddle.org

添加您希望加载和构建内容的目录列表(选项包括：--paddle，--book，--models，--mobile)

运行：

```
./runserver --paddle <path_to_FluidDoc_dir>
```

**注意：**  `<pathe_to_FluidDoc_dir>`为第一步中paddle副本在您本机的存储地址。

如果您需要处理依赖于`book`、`models`或`mobile`存储库内容的文档，您可以添加一个或多个可选项：

```
./runserver --paddle <path_to_fluiddoc_dir> \
    --book <path_to_fluiddoc_dir>/external/book \
    --models <path_to_fluiddoc_dir>/external/models \
    --mobile <path_to_fluiddoc_dir>/external/mobile
```
然后：打开浏览器并导航到http://localhost:8000。

>*网站可能需要几秒钟才能成功加载，因为构建需要一定的时间*

>*如果您是在docker环境下运行的这些步骤，请检查ip确保可以将端口8000映射到您的主机*

## 贡献新文档或更新API

所有内容都应该以[Markdown](https://guides.github.com/features/mastering-markdown/) (GitHub风格)的形式编写(尽管在文档中有一些使用.rst格式的遗留内容)。


在完成安装步骤后，您还需要完成下列操作：

  - 在你开始写作之前，我们建议你回顾一下这些关于贡献内容的指南

 ---

  **贡献新文档**


  - 创建一个新的` .md` 文件或者在您当前操作的仓库中修改已存在的文章
  - 将新增的文档名，添加到对应的index文件中

 ---

  **贡献或修改Python API**


  在编译代码的docker容器内,或主机的对应位置：

  - 运行脚本 `paddle/scripts/paddle_build.sh`(在 Paddle repo 下)

  ```bash
  # 编译paddle的python库
  cd Paddle
  ./paddle/scripts/paddle_docker_build.sh gen_doc_lib full
  cd ..
  ```

  - 运行预览工具

  ```
  # 在编译paddle的对应docker镜像中运行预览工具

  docker run -it -v /Users/xxxx/workspace/paddlepaddle_workplace:/workplace -p 8000:8000 [images_id] /bin/bash
  ```

  > 其中`/Users/xxxx/workspace/paddlepaddle_workplace`请替换成您本机的paddle工作环境，`/workplace`请替换成您相应的 docker 下的工作环境，这一映射会保证我们同时完成编译python库、修改FluidDoc和使用预览工具。

  > [images_id]为docker中您使用的paddlepaddle的镜像id。

  - 设置环境变量

  ```
  # 在docker环境中
  # 设置环境变量`PYTHONPATH`使预览工具可以找到 paddle 的 python 库
  export PYTHONPATH=/workplace/Paddle/build/python/
  ```

  - 清理旧文件

  ```
  # 清除历史生成的文件，如果是第一次使用预览工具可以跳过这一步
  rm -rf /workplace/FluidDoc/doc/fluid/menu.json /workplace/FluidDoc/doc/fluid/api/menu.json /tmp/docs/ /tmp/api/
  ```

  - 启动预览工具

  ```
  cd /workplace/PaddlePaddle.org/portal
  pip install -r requirements.txt
  ./runserver --paddle /workplace/FluidDoc/
  ```

---

  **预览修改**



  打开浏览器并导航到http://localhost:8000。

  在要更新的页面上，单击右上角的Refresh Content

  进入使用文档单元后，API部分并不包含内容，希望预览API文档需要点击API目录，几分钟后您将看到生成的 API reference。


## 提交修改

如果您希望修改代码，请在`Paddle`仓库下参考[如何贡献代码](../development/contribute_to_paddle/index_cn.html)执行操作。

如果您仅修改文档：

  - 修改的内容在`doc`文件夹内，您只需要在`FluidDoc`仓库下提交`PR`

  - 修改的内容在`external`文件夹内：

    1.在您修改的仓库下提交PR。这是因为：`FluidDoc`仓库只是一个包装器，将其他仓库的链接（git术语的“submodule”）集合在了一起。

    2.当您的修改被认可后，更新FluidDoc中对应的`submodule`到源仓库最新的commit-id。

      > 例如，您更新了book仓库中的develop分支下的文档：


      > - 进入`FluidDoc/external/book`目录
      > - 更新 commit-id 到最新的提交：`git pull origin develop`
      > - 在`FluidDoc`中提交你的修改

	3.在`FluidDoc`仓库下为您的修改提交PR

提交修改与PR的步骤可以参考[如何贡献代码](../development/contribute_to_paddle/index_cn.html)

## 帮助改进预览工具

我们非常欢迎您对平台和支持内容的各个方面做出贡献，以便更好地呈现这些内容。您可以Fork或Clone这个存储库，或者提出问题并提供反馈，以及在issues上提交bug信息。详细内容请参考[开发指南](https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/DEVELOPING.md)。

## 版权和许可
PaddlePaddle.org在Apache-2.0的许可下提供。
