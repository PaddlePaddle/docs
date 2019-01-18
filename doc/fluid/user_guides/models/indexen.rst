`Fluid Model Library <https://github.com/PaddlePaddle/models/tree/develop/fluid>`__
============

Image classification
--------

Image classification is based on the semantic information of images to distinguish different types of images. It is an important basic problem in computer vision. It is the basis of other high-level visual tasks such as object detection, image segmentation, object tracking, behavior analysis, face recognition, etc. The field has a wide range of applications. Such as: face recognition and intelligent video analysis in the security field, traffic scene recognition in the traffic field, content-based image retrieval and automatic classification of albums in the Internet field, image recognition in the medical field.

In the era of deep learning, the accuracy of image classification has been greatly improved. In the image classification task, we introduced how to train commonly used models in the classic dataset ImageNet, including AlexNet, VGG, GoogLeNet, ResNet, Inception- V4, MobileNet, DPN (Dual
Path Network), SE-ResNeXt model, also open source \ `training model <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/image_classification/README_cn.md#>`__\  is convenient for users to download and use. It also provides tools to convert Caffe models into PaddlePaddle Fluid model configurations and parameter files.

- `AlexNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `VGG <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `GoogleNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `Residual Network <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `Inception-v4 <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `MobileNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `Dual Path Network <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `SE-ResNeXt <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models>`__
- `Convert Caffe model to Paddle Fluid configuration and model file tools <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/caffe2fluid>`__

Target Detection
--------

The goal of the target detection task is to give an image or a video frame, let the computer find the location of all the targets, and give the specific category of each target. For humans, target detection is a very simple task. However, the computer can "see" the number after the image is encoded. It is difficult to solve the high-level semantic concept such as human or object in the image or video frame, and it is more difficult to locate which area of ??the image the target appears in the image. . At the same time, because the target will appear anywhere in the image or video frame, the shape of the target is ever-changing, and the background of the image or video frame varies widely. Many factors make the target detection a challenging problem for the computer. .

In the target detection task, we introduced how to \ `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__\ , \ `MS COCO <http://cocodataset. Org/#home>`__\ Data training general object detection model, currently introduced SSD algorithm, SSD full name Single Shot MultiBox Detector, is one of the newer and better detection algorithms in the target detection field, with fast detection speed and detection High precision.

Detecting faces in an open environment, especially small, obscured and partially occluded faces is also a challenging task. We also introduced how to train Baidu's self-developed face detection PyramidBox model based on `WIDER FACE <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/>`_ data. The algorithm was implemented in March 2018 won the `first name <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html>`_ in the multiple evaluations of WIDER FACE.

- `Single Shot MultiBox Detector <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/object_detection/README_cn.md>`__
- `Face Detector: PyramidBox <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/face_detection/README_cn.md>`_

Image semantic segmentation
------------

Image Semantic Segmentation As the name suggests, image pixels are grouped/segmented according to the semantic meaning of the expression. Image semantics refers to the understanding of image content. For example, it can describe what objects are doing what, etc. Segmentation refers to the image. Each pixel in the label is labeled, and the label belongs to which category. In recent years, it has been used in the driving technology of unmanned vehicles to separate street scenes to avoid pedestrians and vehicles, and auxiliary diagnosis in medical image analysis.

In the image semantic segmentation task, we introduce how to perform semantic segmentation based on Image Cascade Network (ICNet). Compared with other segmentation algorithms, ICNet takes into account the accuracy and speed.

- `ICNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/icnet>`__

Image generation
-----------

Image generation refers to generating a target image based on an input vector. The input vector here can be random noise or a user-specified condition vector. Specific application scenarios include: handwriting generation, face synthesis, style migration, image restoration, and the like. Current image generation tasks are primarily achieved by generating a confrontation network (GAN). The Generated Confrontation Network (GAN) consists of two subnetworks: a generator and a recognizer. The input to the generator is a random noise or condition vector and the output is the target image. The recognizer is a classifier, the input is an image, and the output is whether the image is a real image. During the training process, the generator and the recognizer enhance their abilities through constant mutual gaming.

In the image generation task, we introduced how to use DCGAN and ConditioanlGAN to generate handwritten numbers, and also introduced CycleGAN for style migration.

- `DCGAN & ConditionalGAN <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/gan/c_gan>`__
- `CycleGAN <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/gan/cycle_gan>`__

Scene text recognition
------------

Many scene images contain rich text information, which plays an important role in understanding image information and can greatly help people understand and understand the content of scene images. Scene text recognition is a process of converting image information into a sequence of characters in the case of complex image background, low resolution, diverse fonts, random distribution, etc. It can be considered as a special translation process: translation of image input into natural language output. . The development of scene image text recognition technology has also promoted the emergence of some new applications, such as automatically identifying street signs to help obtain more accurate address information by automatically recognizing text in street signs.

In the scene text recognition task, we introduce how to combine CNN-based image feature extraction and RNN-based sequence translation technology, eliminating artificial definition features, avoiding character segmentation, and using automatically learned image features to complete character recognition. Currently, the CRNN-CTC model and the sequence-to-sequence model based on the attention mechanism are introduced.

- `CRNN-CTC model <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/ocr_recognition>`__
- `Attention Model <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/ocr_recognition>`__


Metric learning
-------


Metric learning is also called distance metric learning and similarity learning. Through the distance between learning objects, metric learning can be used to analyze the association and comparison of object time. It can be applied to practical problems and can be applied to auxiliary classification and aggregation. Class problems are also widely used in areas such as image retrieval and face recognition. In the past, for different tasks, it is necessary to select appropriate features and manually construct a distance function, and the metric learning can learn the metric distance function for a specific task from the main task according to different tasks. The combination of metric learning and deep learning has achieved good performance in the fields of face recognition/verification, human re-ID, image retrieval, etc. In this task, we mainly introduce the depth-based metric learning based on Fluid. The model contains loss functions such as triples and quaternions.

- `Metric Learning <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/metric_learning>`__


Video classification
-------

Video classification is the basis of video understanding tasks. Unlike image classification, classified objects are no longer still images, but a video object composed of multi-frame images containing speech data and motion information, so understand Video needs to get more context information, not only to understand what each frame image is, what it contains, but also to combine the different frames to know the context related information. The video classification method mainly includes a method based on a convolutional neural network, a cyclic neural network, or a combination of the two. In this task, we introduce the Fluid-based video classification model, which currently includes the Temporal Segment Network (TSN) model, which will continue to add more models.


- `TSN <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/video_classification>`__



Speech Recognition
--------

Automatic Speech Recognition (ASR) is a technique for transcribed vocabulary content in human voice into words that can be input by a computer. The research on speech recognition has undergone a long process of exploration. After the HMM/GMM model, its development has been slow. With the rise of deep learning, it has ushered in spring. In the multi-language recognition task, the deep neural network (DNN) is used as an acoustic model to achieve better performance than the GMM, making ASR one of the most successful areas of deep learning applications. Due to the continuous improvement of recognition accuracy, more and more language technology products have been able to land, such as language input methods, smart home devices represented by smart speakers, etc. - language-based interaction is profoundly changing human life.

Different from the end-to-end direct prediction word distribution of the deep learning model in `DeepSpeech <https://github.com/PaddlePaddle/DeepSpeech>`__, this example is closer to the traditional language recognition process, with phoneme as the modeling unit. Focus on the training of acoustic models in speech recognition, use `kaldi <http://www.kaldi-asr.org>`__ for feature extraction and label alignment of audio data, and integrate kaldi's decoder to complete decoding.

- `DeepASR <https://github.com/PaddlePaddle/models/blob/develop/fluid/DeepASR/README_cn.md>`__

machine translation
--------

Machine Translation transforms a natural language (source language) into a natural language (target speech), which is a very basic and important research direction in natural language processing. In the wave of globalization, the important role played by machine translation in promoting cross-language civilization communication is self-evident. Its development has gone through stages such as statistical machine translation and neural network-based neuro-machine translation (NMT). After NMT matured, machine translation was really applied on a large scale. The early stage of NMT is mainly based on the cyclic neural network RNN. The current time step in the training process depends on the calculation of the previous time step, and it is difficult to parallelize the time steps to improve the training speed. Therefore, NMTs of non-RNN structures have emerged, such as structures based on convolutional neural networks CNN and structures based on Self-Attention.

The Transformer implemented in this example is a machine translation model based on the self-attention mechanism, in which there is no more RNN or CNN structure, but the context dependency in the Attention learning language is fully utilized. Compared with RNN/CNN, this structure has lower computational complexity, easier parallelization, and easier modeling for long-range dependencies in a single layer, and finally achieves the best translation effect among multiple languages.


- `Transformer <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/neural_machine_translation/transformer/README_cn.md>`__

Reinforcement learning
--------

Reinforcement learning is an increasingly important machine learning direction in recent years, especially Deep Reinforcement Learning (DRL), which is combined with deep learning, and has achieved many amazing achievements. The well-known AlphaGo, which defeats the top-level Go players, is a typical example of DRL applications. In addition to the game field, other applications include robots, natural language processing and so on.

The pioneering work of deep reinforcement learning is a successful application in Atari video games, which can directly accept high-dimensional input of video frames and predict the next action according to the image content end-to-end. The model used is called depth. Q Network (Deep Q-Network, DQN). This example uses the flexible framework of PaddlePaddle Fluid to implement DQN and its variants and test their performance in Atari games.

- `DeepQNetwork <https://github.com/PaddlePaddle/models/blob/develop/fluid/DeepQNetwork/README_cn.md>`__

Chinese lexical analysis
------------

Word Segmentation is the process of segmenting continuous natural language text into lexical sequences with semantic rationality and integrity. Because in Chinese, words are the most basic unit of semantics, and words are the basis of many natural language processing tasks such as text classification, sentiment analysis, and information retrieval. Part-of-speech Tagging is a process of assigning a part of speech to each vocabulary in a natural language text. The part of speech here includes nouns, verbs, adjectives, adverbs, and so on. Named Entity Recognition (NER), also known as "name identification", refers to the identification of entities with specific meanings in natural language text, including person names, place names, institution names, proper nouns, and so on. We unify these three tasks into a joint task called lexical analysis task, based on deep neural network, using massively labeled corpus for training, providing an end-to-end solution.

We named this joint Chinese lexical analysis solution LAC. LAC can be considered as an acronym for Lexical Analysis of Chinese, or as a recursive abbreviation for LAC Analyzes Chinese.

- `LAC <https://github.com/baidu/lac/blob/master/README.md>`__

Affective tendency analysis
------------

The sentiment orientation analysis is for Chinese text with subjective description, which can automatically judge the emotional polarity category of the text and give corresponding confidence. The types of emotions are divided into positive, negative and neutral. Affective sentiment analysis can help companies understand user spending habits, analyze hot topics and crisis public opinion monitoring, and provide strong decision support for enterprises. This time we open the AI ??open platform to analyze the sentiment orientation using the `model <http://ai.baidu.com/tech/nlp/sentiment_classify>`__, which is available to users.

- `Senta <https://github.com/baidu/Senta/blob/master/README.md>`__

Semantic matching
--------

In many scenarios of natural language processing, it is necessary to measure the semantic similarity of two texts. Such tasks are often called semantic matching. For example, in the search, the search results are sorted according to the similarity between the query and the candidate document, the text deduplicates the calculation of the similarity between the text and the text, and the matching of the candidate answers and the questions in the automatic question and answer.

The DAM (Deep Attention Matching Network) opened in this example is the work of Baidu Natural Language Processing Department published in ACL-2018, which is used for the selection of responses in multi-round dialogue of search chat robots. Inspired by Transformer, DAM is based entirely on the attention mechanism. It uses the stack-type self-attention structure to learn the semantic representations of responses and contexts at different granularities, and then uses cross-attention to obtain responses and contexts. The correlation between the two large-scale multi-round dialogue data sets is better than other models.

- `Deep Attention Matching Network <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleNLP/deep_attention_matching_net>`__

AnyQ
----

`AnyQ <https://github.com/baidu/AnyQ>`__\ (ANswer Your Questions)
The open source project mainly includes a question and answer system framework for the FAQ collection and a text semantic matching tool SimNet. The Q&A system framework adopts a configuration and plug-in design. Each function is added through a plug-in form. Currently, 20+ plug-ins are open. Developers can use the AnyQ system to quickly build and customize FAQ Q&A systems for specific business scenarios and accelerate iterations and upgrades.

SimNet is a semantic matching framework independently developed by Baidu's Natural Language Processing Department in 2013. The framework is widely used in Baidu's products, including core network structures such as BOW, CNN, RNN, and MM-DNN, and is also integrated based on the framework. The mainstream semantic matching models in the academic world, such as MatchPyramid, MV-LSTM, K-NRM and other models. Models built using SimNet can be easily added to the AnyQ system to enhance the semantic matching capabilities of the AnyQ system.


- `SimNet in PaddlePaddle Fluid <https://github.com/baidu/AnyQ/blob/master/tools/simnet/train/paddle/README.md>`__

Machine reading comprehension
----

Machine Reading Comprehension (MRC) is one of the core tasks in Natural Language Processing (NLP). The ultimate goal is to let machines read texts like humans, refine text information and answer related questions. Deep learning has been widely used in NLP in recent years, and the machine reading comprehension ability has been greatly improved in recent years. However, the machine reading comprehension of the current research uses artificially constructed data sets, and answers some relatively simple questions, and human processing. There is still a clear gap in the data, so there is an urgent need for large-scale real training data to promote the further development of MRC.

Baidu reading comprehension data set is a real-world data set open sourced by Baidu Natural Language Processing Department. All the questions and original texts are derived from actual data (Baidu search engine data and Baidu know Q&A community), and the answer is answered by humans. Each question corresponds to multiple answers. The data set contains 200k questions, 1000k original text and 420k answers. It is currently the largest Chinese MRC data set. Baidu also open sourced the corresponding reading comprehension model, called DuReader, using the current common network hierarchical structure, capturing the interaction between the problem and the original text through the two-way attention mechanism, generating the original representation of the query-aware, and finally based on the query-aware The original text indicates that the answer network is predicted by the point network.

- `DuReader in PaddlePaddle Fluid <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/machine_reading_comprehension/README.md>`__


Personalized recommendation
-------

The recommendation system is playing an increasingly important role in the current Internet service. At present, most e-commerce systems, social networks, advertisement recommendation, and search engines all use various forms of personalized recommendation technology to help users. Quickly find the information they want.

In an industrially available recommendation system, the recommendation strategy is generally divided into multiple modules in series. Take the news recommendation system as an example. There are multiple links that can use deep learning techniques, such as automated annotation of news, personalized news recall, personalized matching and sorting. PaddlePaddle provides complete support for the training of recommended algorithms and provides a variety of model configurations for users to choose from.

- `TagSpace <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/tagspace>`_
- `GRU4Rec <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/gru4rec>`_
- `SequenceSemanticRetrieval <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/ssr>`_
- `DeepCTR <https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/ctr/README.cn.md>`_
- `Multiview-Simnet <https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/multiview_simnet>`_
