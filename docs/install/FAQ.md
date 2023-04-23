<a name="FAQ"></a>
# **FAQ**

- 报错“nccl.h 找不到”

    > 请[安装 nccl2](https://developer.nvidia.com/nccl/nccl-download)

<!--TODO please add more F&Q parts here-->

- Ubuntu18.04 下 libidn11 找不到？

    > 使用以下指令：

        apt install libidn11

- Ubuntu 编译时出现大量的代码段不能识别？

    > 这可能是由于 cmake 版本不匹配造成的，请在 gcc 的安装目录下使用以下指令：

        apt install gcc-4.8 g++-4.8
        cp gcc gcc.bak
        cp g++ g++.bak
        rm gcc
        rm g++
        ln -s gcc-4.8 gcc
        ln -s g++-4.8 g++

- 遇到 paddlepaddle.whl is not a supported wheel on this platform？

    > 出现这个问题的主要原因是，没有找到和当前系统匹配的 paddlepaddle 安装包。 请检查 Python 版本是否为 2.7 系列。另外最新的 pip 官方源中的安装包默认是 manylinux1 标准， 需要使用最新的 pip (>9.0.0) 才可以安装。您可以执行以下指令更新您的 pip：

        pip install --upgrade pip
    或者

        python -c "import pip; print(pip.pep425tags.get_supported())"

    > 如果系统支持的是 linux_x86_64 而安装包是 manylinux1_x86_64 ，需要升级 pip 版本到最新； 如果系统支持 manylinux1_x86_64 而安装包     （本地）是 linux_x86_64， 可以重命名这个 whl 包为 manylinux1_x86_64 再安装。

- 使用 Docker 编译出现问题？

    > 请参照 GitHub 上[Issue12079](https://github.com/PaddlePaddle/Paddle/issues/12079)

- 可以用 IDE 吗？

      > 当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。
      很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行
      `global-set-key "\C-cc" 'compile`
      `setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev"`
      就可以按 `Ctrl-C` 和 `c` 键来启动编译了。

- 可以并行编译吗？

      > 是的。我们的 Docker image 运行一个 [Bash 脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/paddle_build.sh)。这个脚本调用`make -j$(nproc)` 来启动和 CPU 核一样多的进程来并行编译。

- 在 Windows/macOS 上编译很慢？

      > Docker 在 Windows 和 macOS 都可以运行。不过实际上是运行在一个 Linux 虚拟机上。可能需要注意给这个虚拟机多分配一些 CPU 和内存，以保证编译高效。具体做法请参考[issue627](https://github.com/PaddlePaddle/Paddle/issues/627)。

- 磁盘不够？

      > 本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。`docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。可以参考 [这篇文章](https://zaiste.net/posts/removing_docker_containers) 来清理这些内容。

- 在 DockerToolbox 下使用 book 时`http://localhost:8888/`无法打开？

       > 需要将 localhost 替换成虚拟机 ip，一般需要在浏览器中输入：`http://192.168.99.100:8888/`

- pip install gpu 版本的 PaddlePaddle 后运行出现 SegmentFault 如下：

       @ 0x7f6c8d214436 paddle::platform::EnforceNotMet::EnforceNotMet()

        @ 0x7f6c8dfed666 paddle::platform::GetCUDADeviceCount()

       @ 0x7f6c8d2b93b6 paddle::framework::InitDevices()


       > 出现这个问题原因主要是由于您的显卡驱动低于对应 CUDA 版本的要求，请保证您的显卡驱动支持所使用的 CUDA 版本

<a name="OPENBLAS"></a>

- macOS 下使用自定义的 openblas 详见 issue：

      > [ISSUE 13217](https://github.com/PaddlePaddle/Paddle/issues/13721)

- 已经安装 swig 但是仍旧出现 swig 找不到的问题 详见 issue：

    >  [ISSUE 13759](https://github.com/PaddlePaddle/Paddle/issues/13759)

- 出现 “target pattern contain no '%'.”的问题 详见 issue：

    > [ISSUE 13806](https://github.com/PaddlePaddle/Paddle/issues/13806)
