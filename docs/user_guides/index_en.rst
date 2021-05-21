###########
User Guides
###########

..  todo::

If you have got the hang of Beginner's Guide, and wish to model practical problems and build your original networks, this section will provide
you with some detailed operations:


This section collects several documents arranging from the simplest to the most challenging, which will guide you through the basic deep learning tasks in PaddlePaddle.

The documentation in this chapter covers a lot of deep learning basics and how to implement them with PaddlePaddle. See the instructions below for how to use:


Overview
======================

   
    - `Simple Case <../user_guides/simple_case/index_en.html>`_ ：introduces basic cases of Paddle

    - `Computer Vision <../user_guides/cv_case/index_en.html>`_ ：introduces cases of using paddle to realize Computer Vision task

    - `Natural Language Processing <../user_guides/nlp_case/index_en.html>`_：introduces cases of using paddle to realize Natural Language Processing tasks

    - `Recommend <../user_guides/rec_case/index_en.html>`_：introduces cases of using paddle to realize Recommend tasks

	- `Models Zoo <../user_guides/models/index_en.html>`_：introduces the models zoo of Paddle

..  toctree::
    :hidden:

    simple_case/index_en.rst
    cv_case/index_en.rst
    nlp_case/index_en.rst
    rec_case/index_en.rst
	models/index_cn.rst


We packaged Jupyter, PaddlePaddle, and various dependency softwares into a Docker image. It frees you from installing these softwares by yourself, and you only need to just install Docker. For various Linux versions, please refer to https://www.docker.com . If you use docker on `Windows <https://www.docker.com/docker-windows>`_ or `Mac <https://www.docker.com/docker-mac>`_ , consider `allocate more Memory and CPU resources to Docker  <http://stackoverflow.com/a/39720010/724872>`_ .


Instructions
======================


This book assumes you are performing CPU training by default. If you want to use GPU training, the steps will vary slightly. Please refer to "GPU Training" below.





CPU training
>>>>>>>>>>>>

Just run these in shell:

..  code-block:: shell

	docker run -d -p 8888:8888 paddlepaddle/book

It downloads the Docker image for running books from DockerHub.com.
To read and edit this book on-line, please visit http://localhost:8888 in your browser.

If the Internet connection to DockerHub.com is compromised, try our spare docker image named docker.paddlepaddlehub.com:

::

	docker run -d -p 8888:8888 docker.paddlepaddlehub.com/book


GPU training
>>>>>>>>>>>>>

To ensure that the GPU driver works properly in the image, we recommend running the image with `nvidia docker <https://github.com/NVIDIA/nvidia-docker>`_ . Please install nvidia-docker first, then run:


::

	nvidia-docker run -d -p 8888:8888 paddlepaddle/book:latest-gpu


Or use a image source in China to run:

::

	nvidia-docker run -d -p 8888:8888 docker.paddlepaddlehub.com/book:latest-gpu


modify the following codes

..  code-block:: python

	use_cuda = False


into :

..  code-block:: python

	use_cuda = True



Contribute to Book
===================

We highly appreciate your original contributions of new chapters to Book! Just Pull Requests of your contributions to the sub-directory in :code:`pending` . When this chapter is endorsed, we'll gladly move it to the root directory.


For writing, running, debugging, you need to install `shell <https://github.com/PaddlePaddle/book/blob/develop/.tools/convert-markdown-into-ipynb-and-test.sh>`_ to generate Docker image。

**Please Note:** We also provide `English Readme <https://github.com/PaddlePaddle/book/blob/develop/README.md>`_ for PaddlePaddle book
