# How to contribute documentation

PaddlePaddle encourages you to contribute documentation. If your written or translated documents meet our requirements, your documents will be available on the paddlapaddle.org website and on Github for PaddlePaddle users.

Paddle's documentation is mainly divided into the following modules:

- Beginner's Guide: It includes installation instructions, basic knowledge of deep learning, learning materials, etc., designed to help users get started and inspired;

- User Guides: It includes data preparation, network configuration, training, debug, predictive deployment, and model library to provide users with a tutorial of basic operations in PaddlePaddle;

- Advanced User Guides: It includes server-side and mobile-end deployment, how to contribute code or documentation, how to optimize performance, etc., designed to meet the needs of developers;

Our documentation supports contributions in format of [reStructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) and [Markdown](https://guides.github.com/features/ Mastering-markdown/) (GitHub style) .

Once the document is written, you can use the preview tool to check how the document appears on the official website to verify that your document is displayed correctly on the official website.


## How to use the preview tool

If you are modifying a code document (i.e. API) and using PaddlePaddle in a Docker container, perform the following steps in your corresponding docker container. Because the API's document generator relies on PaddlePaddle.

If you have only improved text or media content (you don't need to install or build PaddlePaddle), or are building PaddlePaddle on the host machine, please continue with the following steps on the host.

### 1. Clone related repository that you wish to update or test:

First download the full document repository, where `--recurse-submodules` will update the submodules in FluidDoc (all submodules are in `FluidDoc/external`) to ensure that all documents will be displayed properly:

```
git clone --recurse-submodules https://github.com/PaddlePaddle/FluidDoc
```

Other pullable repositories are:


```
git clone https://github.com/PaddlePaddle/book.git
git clone https://github.com/PaddlePaddle/models.git
git clone https://github.com/PaddlePaddle/Mobile.git

```

You can place these local copies in any directory on your computer, and we will specify the location of these repositories when we start PaddlePaddle.org.

### 2. Pull PaddlePaddle.org in a new directory and install its dependencies

Before doing this, please make sure your operating system has python dependencies installed.

Take the ubuntu system as an example, run:

```
sudo apt-get update && apt-get install -y python-dev build-essential
```

Then:

```
git clone https://github.com/PaddlePaddle/PaddlePaddle.org.git
cd PaddlePaddle.org/portal
```

Then install requirements. Please  make sure that you install with python 2.7.15 or 2.7.16. We recommend you to use Anaconda or virtualenv to create an appropriate virtual environment first. 

Install requirements：


```
pip install -r requirements.txt
```

**Optional**: If you wish to implement a Chinese-English website conversion to improve PaddlePaddle.org, please install [GNU gettext](https://www.gnu.org/software/gettext/)

### 3. Run PaddlePaddle.org locally

Add a list of options which you want to load and build (options include: --paddle, --book, --models, --mobile)

run:

```
./runserver --paddle <path_to_FluidDoc_dir>
```

**Note:** `<path_to_FluidDoc_dir>` is the directory of the local FluidDoc copy specified in the first step.

If you need to work with documents that depend on the contents of the `book`, `models`, or `mobile` repositories, you can add one or more options:

```
./runserver --paddle <path_to_fluiddoc_dir> \
	--book <path_to_fluiddoc_dir>/external/book \
	--models <path_to_fluiddoc_dir>/external/models \
	--mobile <path_to_fluiddoc_dir>/external/mobile
```
Then: open your browser and navigate to http://localhost:8000.

>* The site may take a few seconds to load because the building takes a certain amount of time*

>* If you are running these steps in a docker environment, please check ip to make sure port 8000 can be mapped to your host*

## Contribute new documentation or update API

All content should be written in [Markdown](https://guides.github.com/features/mastering-markdown/) (GitHub style) (although there are some legacy content with the .rst format in the documentation).


After completing the installation, you will also need to do:

  - Before you start writing, we suggest that you review the following tips for contributing content.

 ---

  **Contribute new documents**


  - Create a new `.md` file or modify an existing article in the repository you are currently working on
  - Add the new document name to the corresponding index file

 ---

  **Contribute or modify the Python API**


  In the docker container that compiles the code, or the corresponding location in the host machine:

  - Run the script `paddle/scripts/paddle_build.sh` (under Paddle repo)

  ```bash
  # Compile paddle's python library
  cd Paddle
  ./paddle/scripts/paddle_docker_build.sh gen_doc_lib full
  cd ..
  ```

  - Run the preview tool

  ```
  # Run the preview tool in docker image which compiled paddle

  docker run -it -v /Users/xxxx/workspace/paddlepaddle_workplace:/workplace -p 8000:8000 [images_id] /bin/bash
  ```

  > Where `/Users/xxxx/workspace/paddlepaddle_workplace` should be replaced with your local host paddle workspace directory, `/workplace` should be replaced with the working directory in the docker. This mapping will ensure that we compile the python library, modify FluidDoc and use the preview tool at the same time.

  > [images_id] is the image id of the paddlepaddle you use in docker.

  - Set environment variables

  ```
  # In docker environment
  # Set the environment variable `PYTHONPATH` so that the preview tool can find the python library for paddle
  export PYTHONPATH=/workplace/Paddle/build/python/
  ```

  - Clean up old files

  ```
  # Clear the previously generated file, if you are using the preview tool for the first time, you can skip this step
  rm -rf /workplace/FluidDoc/doc/fluid/menu.json /workplace/FluidDoc/doc/fluid/api/menu.json /tmp/docs/ /tmp/api/
  ```

  - Launch preview tool

  ```
  cd /workplace/PaddlePaddle.org/portal
  pip install -r requirements.txt
  ./runserver --paddle /workplace/FluidDoc/
  ```

---

  **Preview modification**



  Open your browser and navigate to http://localhost:8000 .

  On the page to be updated, click Refresh Content at the top right corner.

  After entering documentation page, the API section does not contain content. To preview the API document, please click on the API directory and you will see the generated API reference after a few minutes.


## Submit changes

If you wish to modify the code, please refer to [How to contribute code](../development/contribute_to_paddle/index_en.html) under the `Paddle` repository.

If you just modify the document:

  - The modified content is in the `doc` folder, you only need to submit `PR` in the `FluidDoc` repository.

  - The modified content is in the `external` folder:

	1. Submit the PR in the repostory you modified. This is because the `FluidDoc` repository is just a wrapper that brings together the links of other repositories (namely, the "submodules" in git context).

	2. When your changes are approved, update the corresponding `submodule` in FluidDoc to the latest commit-id of the source repository.

	  > For example, you updated the document on the develop branch in the book repository:


	  > - Go to the `FluidDoc/external/book` directory
	  > - Update commit-id to the latest commit: `git pull origin develop`
	  > - Commit your changes in `FluidDoc`

3.  Pull Request for your changes in the `FluidDoc` repository

The steps to submit changes and PR can refer to [How to contribute code](../development/contribute_to_paddle/index_en.html)

## Help improve preview tool

We encourage your contributions to all aspects of the platform and supportive contents. You can Fork or Clone repository, ask questions and feedback, or submit bugs on issues. For details, please refer to the [Development Guide](https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/DEVELOPING.md).

## Copyright and Licensing
PaddlePaddle.org is available under the Apache-2.0 license.
