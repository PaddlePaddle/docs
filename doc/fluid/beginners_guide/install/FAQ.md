***
<a name="FAQ"></a>        
# **FAQ**
- CentOS6下如何编译python2.7为共享库? 
	
	> 使用以下指令：
	
		./configure --prefix=/usr/local/python2.7 --enable-shared   
		make && make install   

<!--TODO please add more F&Q parts here-->

- Ubuntu18.04下libidn11找不到？
	
	> 使用以下指令：
	
		apt install libidn11   

- Ubuntu编译时出现大量的代码段不能识别？
	
	> 这可能是由于cmake版本不匹配造成的，请在gcc的安装目录下使用以下指令：
		
		apt install gcc-4.8 g++-4.8
		cp gcc gcc.bak
		cp g++ g++.bak
		rm gcc
		rm g++
		ln -s gcc-4.8 gcc
		ln -s g++-4.8 g++
        



- 遇到paddlepaddle*.whl is not a supported wheel on this platform？
	> 出现这个问题的主要原因是，没有找到和当前系统匹配的paddlepaddle安装包。 请检查Python版本是否为2.7系列。另外最新的pip官方源中的安装包默认是manylinux1标准， 需要使用最新的pip (>9.0.0) 才可以安装。您可以执行以下指令更新您的pip：     
	
	pip install --upgrade pip     

	> 或者：     
	
	python -c "import pip; print(pip.pep425tags.get_supported())"    

	> 如果系统支持的是 linux_x86_64 而安装包是 manylinux1_x86_64 ，需要升级pip版本到最新； 如果系统支持 manylinux1_x86_64 而安装包	 （本地）是 linux_x86_64， 可以重命名这个whl包为 manylinux1_x86_64 再安装。

- 使用Docker编译出现问题？
	
	> 请参照GitHub上[Issue12079](https://github.com/PaddlePaddle/Paddle/issues/12079)

- 什么是 Docker?

  如果您没有听说 Docker，可以把它想象为一个类似 virtualenv 的系统，但是虚拟的不仅仅是 Python 的运行环境。

- Docker 还是虚拟机？

  有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。

- 为什么用 Docker?

  把工具和配置都安装在一个 Docker image 里可以标准化编译环境。这样如果遇到问题，其他人可以复现问题以便帮助。

  另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。

- 可以选择不用Docker吗？

  当然可以。大家可以用把开发工具安装进入 Docker image 一样的方式，把这些工具安装到本机。这篇文档介绍基于 Docker 的开发流程，是因为这个流程比其他方法都更简便。

- 学习 Docker 有多难？

  理解 Docker 并不难，大概花十分钟看一下[这篇文章](https://zhuanlan.zhihu.com/p/19902938)。
  这可以帮您省掉花一小时安装和配置各种开发工具，以及切换机器时需要新安装的辛苦。别忘了 PaddlePaddle 更新可能导致需要新的开发工具。更别提简化问题复现带来的好处了。

- 可以用 IDE 吗？

  当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。

  很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行

    (global-set-key "\C-cc" 'compile)
    (setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")

  就可以按 `Ctrl-C` 和 `c` 键来启动编译了。

- 可以并行编译吗？

  是的。我们的 Docker image 运行一个 [Bash 脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/paddle/scripts/paddle_build.sh)。这个脚本调用`make -j$(nproc)` 来启动和 CPU 核一样多的进程来并行编译。

- Docker 需要 sudo？

  如果用自己的电脑开发，自然也就有管理员权限（sudo）了。如果用公用的电脑开发，需要请管理员安装和配置好 Docker。此外，PaddlePaddle 项目在努力开始支持其他不需要 sudo 的集装箱技术，比如 rkt。

- 在 Windows/MacOS 上编译很慢？

  Docker 在 Windows 和 MacOS 都可以运行。不过实际上是运行在一个 Linux 虚拟机上。可能需要注意给这个虚拟机多分配一些 CPU 和内存，以保证编译高效。具体做法请参考[issue627](https://github.com/PaddlePaddle/Paddle/issues/627)。

- 磁盘不够？

  本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。`docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。可以参考 [这篇文章](https://zaiste.net/posts/removing_docker_containers) 来清理这些内容。

- 在DockerToolbox下使用book时`http://localhost:8888/`无法打开？
	
   需要将localhost替换成虚拟机ip，一般需要在浏览器中输入：`http://192.168.99.100:8888/`

- pip install gpu版本的PaddlePaddle后运行出现SegmentFault如下：
   
  	 @ 0x7f6c8d214436 paddle::platform::EnforceNotMet::EnforceNotMet()
	 
   	 @ 0x7f6c8dfed666 paddle::platform::GetCUDADeviceCount() 
	 
  	 @ 0x7f6c8d2b93b6 paddle::framework::InitDevices()
   
   出现这个问题原因主要是由于您的显卡驱动低于对应CUDA版本的要求，请保证您的显卡驱动支持所使用的CUDA版本
   

<a name="MACPRO"></a>    

- MacOS下安装PaddlePaddle后import paddle.fluid出现`Fatal Python error: PyThreadState_Get: no current thread running`错误 

	请使用`find / -name libpython2.7.dylib`找到您当前使用python的`libpython2.7.dylib`路径，并使用`export LD_LIBRARY_PATH=[libpython2.7.dylib所在的目录的上两级目录] && export DYLD_LIBRARY_PATH=[libpython2.7.dylib所在的目录的上两级目录]`
