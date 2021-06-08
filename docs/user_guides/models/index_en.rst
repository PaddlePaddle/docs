`Fluid Model Library <https://github.com/PaddlePaddle/models>`__
============

Image classification
--------

Image classification is based on the semantic information of images to distinguish different types of images. It is an important basic problem in computer vision. It is the basis of other high-level visual tasks such as object detection, image segmentation, object tracking, behavior analysis, face recognition, etc. The field has a wide range of applications. Such as: face recognition and intelligent video analysis in the security field, traffic scene recognition in the traffic field, content-based image retrieval and automatic classification of albums in the Internet field, image recognition in the medical field.

In the era of deep learning, the accuracy of image classification has been greatly improved. In the image classification task, we introduced how to train commonly used models in the classic dataset ImageNet, including AlexNet, VGG, GoogLeNet, ResNet, Inception- V4, MobileNet, DPN (Dual
Path Network), SE-ResNeXt model. We also provide open source \ `trained model <https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_cn.md#>`__\  to make it convenient for users to download and use. It also provides tools to convert Caffe models into PaddlePaddle Fluid model configurations and parameter files.

- `AlexNet <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `VGG <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `GoogleNet <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `Residual Network <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `Inception-v4 <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `MobileNet <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `Dual Path Network <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `SE-ResNeXt <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models>`__
- `Convert Caffe model to Paddle Fluid configuration and model file tools <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/caffe2fluid>`__

Object Detection
-----------------

The goal of the object detection task is to give an image or a video frame, let the computer find the locations of all the objects, and give the specific category of each object. For humans, target detection is a very simple task. However, the computer can only "see" the number after the image is encoded. It is difficult to solve the high-level semantic concept such as human or object in the image or video frame, and it is more difficult to locate the area where the target appears in the image. At the same time, because the target will appear anywhere in the image or video frame, the shape of the target is ever-changing, and the background of the image or video frame varies widely. Many factors make the object detection a challenging problem for the computer.

In the object detection task, we introduced how to train general object detection model based on dataset  `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__\ , \ `MS COCO <http://cocodataset. Org/#home>`__\ . Currently we introduced SSD algorithm, which is the acronym for Single Shot MultiBox Detector. As one of the newer and better detection algorithms in the object detection field, it features fast detection speed and detection High precision.

Detecting human faces in an open environment, especially small, obscured and partially occluded faces is also a challenging task. We also introduced how to train Baidu's self-developed face detection PyramidBox model based on `WIDER FACE <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/>`_ data. The algorithm won the `first place <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html>`_ in multiple evaluations of WIDER FACE in March 2018 .

- `Single Shot MultiBox Detector <https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/README_cn.md>`__
- `Face Detector: PyramidBox <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/face_detection/README_cn.md>`_

Image semantic segmentation
----------------------------

As the name suggests, Image Semantic Segmentation is to group/segment pixels according to their different semantic meanings. Image semantics refer to the understanding of image content. For example, it can describe what objects are doing what at what location, etc. Segmentation means each pixel in the image is labeled with its category. In recent years, it has been recently used by the driverless vehicles to segment street scenes to avoid pedestrians and vehicles, and by auxiliary diagnosis in medical image analysis.

In the image semantic segmentation task, we introduce how to perform semantic segmentation based on Image Cascade Network (ICNet). Compared with other segmentation algorithms, ICNet takes into account the accuracy and speed.

- `ICNet <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/icnet>`__

Image Synthesis
-----------------

Image Synthesis refers to generating a target image based on an input vector. The input vector here can be random noise or a user-specified condition vector. Specific application scenarios include: handwriting generation, face synthesis, style migration, image restoration, and the like. Current image generation tasks are primarily achieved by Generative Adversarial Networks (GAN). The GAN consists of two subnetworks: a generator and a discriminator. The input to the generator is a random noise or condition vector and the output is the target image. The discriminator is a classifier, the input is an image, and the output is whether the image is a real image. During the training process, the generator and the discriminator enhance their abilities through constant mutual adversarial process.

In the image synthesis task, we introduced how to use DCGAN and ConditioanlGAN to generate handwritten numbers, and also introduced CycleGAN for style migration.

- `DCGAN & ConditionalGAN <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/gan/c_gan>`__
- `CycleGAN <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/gan/cycle_gan>`__

Scene Text Recognition
-----------------------

Rich textual information is usually contained in scene images, which plays an important role in comprehending the image information and greatly helps people recognize and understand the content of images in real scene. Text recognition in real scene images is a process of converting image information into a sequence of characters in the case of complex image background, low resolution, diverse fonts, random distribution, etc. It can be considered as a special translation process: translation of image input into natural language output. The development of scene image text recognition technology has also promoted the emergence of some new applications, such as automatically identifying texts in street signs to help street scene applications obtain more accurate address information.

In the scene text recognition task, we introduce how to combine CNN-based image feature extraction and RNN-based sequence translation technology, eliminate artificial definition features, avoid character segmentation, and use automatically learned image features to complete character recognition. Currently, the CRNN-CTC model and the sequence-to-sequence model based on the attention mechanism are introduced.

- `CRNN-CTC model <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition>`__
- `Attention Model <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition>`__


Metric learning
----------------


Metric learning is also called distance metric learning or similarity learning. Through the distance between learning objects, metric learning can be used to analyze the association and comparison of objects. It can be applied to practical problems like auxiliary classification, aggregation and also widely used in areas such as image retrieval and face recognition. In the past, for different tasks, it was necessary to select appropriate features and manually construct a distance function, but the metric learning can initially learn the metric distance function for a specific task from the main task according to different tasks. The combination of metric learning and deep learning has achieved good performance in the fields of face recognition/verification, human re-ID, image retrieval, etc. In this task, we mainly introduce the depth-based metric learning based on Fluid. The model contains loss functions such as triples and quaternions.

- `Metric Learning <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning>`__


Video classification
---------------------

Video classification is the basis of video comprehension tasks. Unlike image classification, classified objects are no longer still images, but a video object composed of multi-frame images containing speech data and motion information, so to understand video needs to get more context information. To be specific, it needs not only to understand what each frame image is, what it contains, but also to combine different frames to know the context related information. The video classification method mainly includes a method based on convolutional neural networks, recurrent neural networks, or a combination of the two. In this task, we introduce the Fluid-based video classification model, which currently includes the Temporal Segment Network (TSN) model, and we will continuously add more models.


- `TSN <https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video_classification>`__



Speech Recognition
--------------------

Automatic Speech Recognition (ASR) is a technique for transcribing vocabulary content in human voice into characters that can be input by a computer. The research on speech recognition has undergone a long term of exploration. After the HMM/GMM model, its development has been relatively slow. With the rise of deep learning, it has come to its spring. In the multi-language recognition task, the deep neural network (DNN) is used as an acoustic model and achieves better performance than the GMM, making ASR one of the most successful fields of deep learning applications. Due to the continuous improvement of recognition accuracy, more and more language technology products have been being implemented, such as language input methods, smart home devices represented by smart speakers, etc. Language-based interaction is profoundly changing our life.

Different from the end-to-end direct prediction for word distribution of the deep learning model  `DeepSpeech <https://github.com/PaddlePaddle/DeepSpeech>`__ , this example is closer to the traditional language recognition process. With phoneme as the modeling unit, it focuses on the training of acoustic models in speech recognition, use `kaldi <http://www.kaldi-asr.org>`__ for feature extraction and label alignment of audio data, and integrate kaldi's decoder to complete decoding.

- `DeepASR <https://github.com/PaddlePaddle/models/blob/develop/PaddleSpeech/DeepASR/README.md>`__

Machine Translation
---------------------

Machine Translation transforms a natural language (source language) into another natural language (target language), which is a very basic and important research direction in natural language processing. In the wave of globalization, the important role played by machine translation in promoting cross-language civilization communication is self-evident. Its development has gone through stages such as statistical machine translation and neural-network-based Neuro Machine Translation (NMT). After NMT matured, machine translation was really applied on a large scale. The early stage of NMT is mainly based on the recurrent neural network RNN. The current time step in the training process depends on the calculation of the previous time step, so it is difficult to parallelize the time steps to improve the training speed. Therefore, NMTs of non-RNN structures have emerged, such as structures based on convolutional neural networks CNN and structures based on Self-Attention.

The Transformer implemented in this example is a machine translation model based on the self-attention mechanism, in which there is no more RNN or CNN structure, but fully utilizes Attention to learn the context dependency. Compared with RNN/CNN, in a single layer, this structure has lower computational complexity, easier parallelization, and easier modeling for long-range dependencies, and finally achieves the best translation effect among multiple languages.


- `Transformer <https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/neural_machine_translation/transformer/README.md>`__

Reinforcement learning
-------------------------

Reinforcement learning is an increasingly important machine learning direction in recent years, and especially Deep Reinforcement Learning (DRL), which combines deep learning and reinforcement learning, has achieved many amazing achievements. The well-known AlphaGo, which defeats the top-level chess players, is a typical example of DRL applications. In addition to the game field, other applications include robots, natural language processing and so on.

The pioneering work of deep reinforcement learning is a successful application in Atari video games, which can directly accept high-dimensional input of video frames and predict the next action according to the image content end-to-end. The model used is called depth Q Network (Deep Q-Network, DQN). This example uses PaddlePaddle Fluid, our flexible framework, to implement DQN and its variants and test their performance in Atari games.

- `DeepQNetwork <https://github.com/PaddlePaddle/models/blob/develop/DeepQNetwork/README_cn.md>`__

Chinese lexical analysis
---------------------------

Word Segmentation is the process of segmenting continuous natural language text into lexical sequences with semantic rationality and integrity. Because in Chinese, word is the most basic unit of semantics, and word segmentation is the basis of many natural language processing tasks such as text classification, sentiment analysis, and information retrieval. Part-of-speech Tagging is a process of assigning a category to each vocabulary in a natural language text. The part of speech category here includes nouns, verbs, adjectives, adverbs, and so on. Named Entity Recognition (NER), also known as "entity name identification", refers to the identification of entities with specific meanings in natural language text, including person names, place names, institution names, proper nouns, and so on. We unify these three tasks into a joint task called lexical analysis task. Based on deep neural network, we use massively labeled corpus for training, and provide an end-to-end solution.

We named this joint Chinese lexical analysis solution LAC. LAC can be considered as an acronym for Lexical Analysis of Chinese, or as a recursive abbreviation for LAC Analyzes Chinese.

- `LAC <https://github.com/baidu/lac/blob/master/README.md>`__

Sentiment analysis
-------------------

The sentiment analysis is for Chinese text with subjective description, which can automatically judge the emotional polarity category of the text and give corresponding confidence. The types of emotions are divided into positive, negative and neutral. Sentiment analysis can help companies understand user spending habits, analyze hot topics and crisis public opinion monitoring, and provide strong decision support for enterprises. This time we publicize the AI open platform to analyze the sentiment orientation using the `model <http://ai.baidu.com/tech/nlp/sentiment_classify>`__, which is available to users.

- `Senta <https://github.com/baidu/Senta/blob/master/README.md>`__

Semantic matching
------------------

In many scenarios of natural language processing, it is necessary to measure the semantic similarity of two texts. Such tasks are often called semantic matching. For example, the search results are sorted according to the similarity between the query and the candidate document; the text deduplication requires the calculation of the similarity between the texts, and the matching of the candidate answers and the questions in the question answering system.

The DAM (Deep Attention Matching Network) introduced in this example is the work of Baidu Natural Language Processing Department published in ACL-2018, which is used for the selection of responses in multi-round dialogue of retrieval chat robots. Inspired by Transformer, DAM is based entirely on the attention mechanism. It uses the stack-type self-attention structure to learn the semantic representations of responses and contexts at different granularities, and then uses cross-attention to obtain relativity between responses and contexts. The performance on the two large-scale multi-round dialogue datasets is better than other models.

- `Deep Attention Matching Network <https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/dialogue_model_toolkit/deep_attention_matching>`__

AnyQ
----

`AnyQ <https://github.com/baidu/AnyQ>`__\ (ANswer Your Questions)
The open source project mainly includes a question and answer system framework for the FAQ collection and a text semantic matching tool SimNet. The Q&A system framework adopts a setting-up manner and plug-in design. Each function is added through a plug-in form. Currently, 20+ plug-ins are open. Developers can use the AnyQ system to quickly build and customize FAQ Q&A systems for specific business scenarios and accelerate iterations and upgrades.

SimNet is a semantic matching framework independently developed by Baidu's Natural Language Processing Department in 2013. The framework is widely used in Baidu's products, including core network structures such as BOW, CNN, RNN, MM-DNN. It is also integrated with mainstream semantic matching model in academic fields based on the framework, such as MatchPyramid, MV-LSTM, K-NRM. Models built by SimNet can be easily added to the AnyQ system to enhance the semantic matching capability of the AnyQ system.


- `SimNet in PaddlePaddle Fluid <https://github.com/baidu/AnyQ/blob/master/tools/simnet/train/paddle/README.md>`__

Machine reading comprehension
---------------------------------

Machine Reading Comprehension (MRC) is one of the core tasks in Natural Language Processing (NLP). The ultimate goal is to let machines read texts like humans, extract text information and answer related questions. Deep learning has been widely used in NLP in recent years, and the machine reading comprehension ability has been greatly improved. However, the machine reading comprehension of the current research uses artificially constructed data sets, and answers some relatively simple questions. There is still a clear gap to the data processed by humans, so there is an urgent need for large-scale real training data to promote the further development of MRC.

Baidu reading comprehension dataset is an open-source real-world dataset publicized by Baidu Natural Language Processing Department. All the questions and original texts are derived from actual data (Baidu search engine data and Baidu know Q&A community), and the answer is given by humans. Each question corresponds to multiple answers. The dataset contains 200k questions, 1000k original text and 420k answers. It is currently the largest Chinese MRC dataset. Baidu also publicized the corresponding open-source reading comprehension model, called DuReader. DuReader adopts the current common network hierarchical structure, and captures the interaction between the problems and the original texts through the double attention mechanism to generate the original representation of the query-aware. Finally, based on the original text of query-aware, the answer scope is predicted by point network.

- `DuReader in PaddlePaddle Fluid <https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/reading_comprehension>`__


Personalized recommendation
---------------------------------

The recommendation system is playing an increasingly important role in the current Internet service. At present, most e-commerce systems, social networks, advertisement recommendation, and search engines all use various forms of personalized recommendation technology to help users quickly find the information they want.

In an industrially adoptable recommendation system, the recommendation strategy is generally divided into multiple modules in series. Take the news recommendation system as an example. There are multiple procedures that can use deep learning techniques, such as automated annotation of news, personalized news recall, personalized matching and sorting. PaddlePaddle provides complete support for the training of recommendation algorithms and provides a variety of model configurations for users to choose from.

- `TagSpace <https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/tagspace>`_
- `GRU4Rec <https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec>`_
- `SequenceSemanticRetrieval <https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ssr>`_
- `DeepCTR <https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/README.cn.md>`_
- `Multiview-Simnet <https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/multiview_simnet>`_
