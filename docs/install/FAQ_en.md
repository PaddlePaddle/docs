***
<a name="FAQ_en"></a>

# **FAQ**

- Ubuntu18.04 under libidn11 can not be found?

    > Use the following instructions:

        apt install libidn11

- When Ubuntu compiles, a lot of code segments are not recognized?

    > This may be caused by a mismatch in the cmake version. Please use the following command in the gcc installation directory:

        apt install gcc-4.8 g++-4.8
        cp gcc gcc.bak
        cp g++ g++.bak
        rm gcc
        rm g++
        ln -s gcc-4.8 gcc
        ln -s g++-4.8 g++




- Encountered paddlepaddle*.whl is not a supported wheel on this platform?

    > The main reason for this problem is that there is no paddlepaddle installation package that matches the current system. Please check if the Python version is 2.7 series. In addition, the latest pip official source installation package defaults to the manylinux1 standard, you need to use the latest pip (>9.0.0) to install. You can update your pip by following these instructions:

        pip install --upgrade pip
    or

        python -c "import pip; print(pip.pep425tags.get_supported())"

    > If the system supports linux_x86_64 and the installation package is manylinux1_x86_64, you need to upgrade the pip version to the latest; if the system supports manylinux1_x86_64 and the installation package (local) is linux_x86_64, you can rename this whl package to manylinux1_x86_64 and install it again.

- Is there a problem with Docker compilation?

    > Please refer to [Issue12079](https://github.com/PaddlePaddle/Paddle/issues/12079) on GitHub.

- What is Docker?

    > If you haven't heard of Docker, you can think of it as a virtualenv-like system, but it virtualises more than the Python runtime environment.

- Is Docker still a virtual machine?

    > Someone uses a virtual machine to analogize to Docker. It should be emphasized that Docker does not virtualize any hardware. The compiler tools running in the Docker container are actually run directly on the native CPU and operating system. The performance is the same as installing the compiler on the machine.

- Why use Docker?

    > Installing the tools and configurations in a Docker image standardizes the build environment. This way, if you encounter problems, others can reproduce the problem to help. In addition, for developers accustomed to using Windows and macOS, there is no need to configure a cross-compilation environment using Docker.

- Can I choose not to use Docker?

    > Of course you can. You can install development tools to the machine in the same way that you install them into Docker image. This document describes the Docker-based development process because it is easier than the other methods.

- How hard is it to learn Docker?

    > It's not difficult to understand Docker. It takes about ten minutes to read this [article](https://zhuanlan.zhihu.com/p/19902938).
    This can save you an hour of installing and configuring various development tools, as well as the need for new installations when switching machines. Don't forget that PaddlePaddle updates may lead to the need for new development tools. Not to mention the benefits of simplifying the recurrence of problems.

- Can I use an IDE?

    > Of course, because the source code is on the machine. By default, the IDE calls a program like make to compile the source code. We only need to configure the IDE to call the Docker command to compile the source code.
    Many PaddlePaddle developers use Emacs. They add two lines to their `~/.emacs` configuration file.
    `global-set-key "\C-cc" 'compile`
    `setq compile-command "docker run --rm -it -v $(git rev-parse --show- Toplevel): /paddle paddle:dev"`
    You can start the compilation by pressing `Ctrl-C` and` c`.

- Can I compile in parallel?

    > Yes. Our Docker image runs a [Bash script](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/paddle_build.sh). This script calls `make -j$(nproc)` to start as many processes as the CPU cores to compile in parallel.

- Docker needs sudo?

    > If you develop with your own computer, you will naturally have admin privileges (sudo). If you are developing from a public computer, you need to ask the administrator to install and configure Docker. In addition, the PaddlePaddle project is working hard to support other container technologies that don't require sudo, such as rkt.

- Is compiling slow on Windows/macOS?

    > Docker runs on both Windows and macOS. However, it is actually running on a Linux virtual machine. It may be necessary to pay attention to allocate more CPU and memory to this virtual machine to ensure efficient compilation. Please refer to [issue627](https://github.com/PaddlePaddle/Paddle/issues/627) for details.

- Not enough disk?

    > In the example in this article, the `--rm` parameter is used in the `docker run`command to ensure that containers after the end of the run are not retained on disk. You can use the `docker ps -a` command to see containers that are stopped but not deleted. The `docker build` command sometimes produces some intermediate results, an image with no name, and it also occupies the disk. You can refer to this [article](https://zaiste.net/removing_docker_containers/) to clean up this content.

- Can't I open `http://localhost:8888/` when using the book under DockerToolbox?

    > You need to replace localhost with virtual machine ip. Generally type this in the browser: `http://192.168.99.100:8888/`

- After the pip install gpu version of PaddlePaddle runing, the SegmentFault appears as follows:

    @ 0x7f6c8d214436 paddle::platform::EnforceNotMet::EnforceNotMet()

    @ 0x7f6c8dfed666 paddle::platform::GetCUDADeviceCount()

    @ 0x7f6c8d2b93b6 paddle::framework::InitDevices()

    > The main reason for this problem is that your graphics card driver is lower than the corresponding CUDA version. Please ensure that your graphics card driver supports the CUDA version used.

- Use customized openblas under macOS. See issue for details:

    >[ISSUE 13217](https://github.com/PaddlePaddle/Paddle/issues/13721)

- Swig has been installed but there is still a problem that swig can't find. See issue for details:

    >[ISSUE 13759](https://github.com/PaddlePaddle/Paddle/issues/13759)

- The question "target pattern contain no '%'." appears. See issue for details:

    >[ISSUE 13806](https://github.com/PaddlePaddle/Paddle/issues/13806)
