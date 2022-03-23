# Educational Neural Network Framework

The purpose of this repository is to serve as a practical material for teaching students the fundamentals of neural
network structure and design.

## Main components

At the moment there are two main components to the repository:

### `nn_lib` package

Contains an implementation of a basic neural network library supporting both forward and backward propagation. The
library is inspired by PyTorch -- a popular ML framework and can be treated as a very simplified version of it. All
operations are essentially performed on NumPy arrays.

For education purposes some methods implementations are removed and students are tasked to implement those methods
themselves. This way the package is only a template of an ML framework. Implementing the missing logic should be a
valuable exersice for the students. On the other hand, the logic that is kept should ease the burden of implementing
everything by themselves and focus students only on the core components responsible for neural network inference and
training.

* `nn_lib.math_fns` implements the expected behaviour of every supported mathematical function during both forward
  (value) and backward (gradient) passes
* `nn_lib.tests` contains rich test base target at checking the correctness of students' implementations
* `nn_lib.tensor` is the core component of `nn_lib`, implements application of math operations on Tensors, and gradient
  propagation and accumulation
* `nn_lib.mdl` contains an interface of a Module class (similar to `torch.nn.Module`) and some implementations of it
* `nn_lib.optim` contains an interface for an NN optimizer and a Stochastic Gradient Descent (SGD) optimizer as the
  simplest version of it
* `nn_lib.data` contains data processing -related components such as Dataset or Dataloader

### `toy_mlp` package

An example usage of `nn_lib` package for the purpose of training a small Multi-Layer Perceptron (MLP) neural network on
a toy dataset of 2D points for binary classification task. Again some methods implementations are removed to be
implemented by students as an exercise.

The example describes a binary MLP NN model (`toy_mlp.binary_mlp_classifier`), a synthetically generated 2D toy
dataset (`toy_mlp.toy_dataset`), a class for training and validating a model (`toy_mlp.model_trainer`) and the main
execution script (`toy_mlp.train_toy_mlp`) that demonstrates a regular pipeline of solving a task using machine learning
approach.

## Setting up

1. Start watching this repository to be notified about updates
2. Clone the repository
3. Create a new private repository for yourself in GitHub and invite me
4. Set up two remotes: the first one for this repo, the second one for your own repo
5. Branch out to `develop` from `master`, commit your changes to `develop` only

## Tasks

### 1. Implementation of `nn_lib` and MLP

Methods marked with a comment `TODO: implement me as an exercise` are for you to implement. Most of the to-implement
functionality is covered by tests inside `nn_lib.tests` directory.

Please note that all the tests should be correct as there exists an implementation that passes all of them. So do not
edit the tests unless you are totally sure and can prove that there is a bug there.

At the end, all the test must pass, but the recommended order of implementation is the following:

1. `.forward()` methods for classes inside `nn_lib.math_fns` (`test_tensor_forward.py`)
2. `.backward()` methods for classes inside `nn_lib.math_fns` (`test_tensor_backward.py`)
3. modules functionality inside `nn_lib.mdl` (`test_modules.py`)
4. optimizers functionality inside `nn_lib.optim` (`test_optim.py`)
5. MLP neural network methods inside `toy_mlp.binary_mlp_classifier.py`
6. training-related methods inside `toy_mlp.model_trainer.py`

If everything is implemented correctly the toy MLP example should be able to be trained successfully reaching 95+ %
validation accuracy (`toy_mlp.train_toy_mlp.py`) on all three of toy datasets.

### 2. Train on MNIST-like dataset

Adapt `toy_mlp` to be trained on some MNIST-like dataset. The main changes would be the following:

- implement `softmax` function inside `nn_lib.tensor_fns.py`
- implement multi-class cross entropy loss at `nn_lib.mdl.ce_loss.py`
- implement a Dataset class for your dataset similarly to `ToyDataset`
    - it is recommended to load datasets from [PyTorch](https://pytorch.org/vision/stable/datasets.html) or
      [TensorFlow](https://www.tensorflow.org/datasets)
    - images will need to be flattened for them to be fed to an MLP
    - labels will need to be converted to one-hot format

You can take any small multiclass image dataset. The examples are the following:
[MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST),
[KMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST),
[QMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.QMNIST.html#torchvision.datasets.QMNIST),
[EMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html#torchvision.datasets.EMNIST),
[FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST)
,
[CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10),
[CIFAR100](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100)
,
[DIGITS](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

### 3. Research task

The final task has some research component to it. In general, you will need to go deeper in one of the areas of machine
learning, perform some experiments and present your findings to the group. The more valuable your findings are, the
better. Examples of tasks are given below, but you are encouraged to suggest your own topics.

- **Confidence map visualization**<br/>
  Implement a code that will visualize binary MLP classifier confidence field for a 2D binary classification dataset,
  for example like [here](http://playground.tensorflow.org/) at the diagram on the right. After that perform experiments
  by varying dataset structure, activation functions, neural network hyperparameters, etc. Look what difference your
  changes make to the diagram and suggest some dos and don'ts based on that.
- **Image embedding space visualization**<br/>
  In machine learning, a term *embedding* usually refers to a low-dimensional representation of a high-dimensional
  object that is learned by a model. For example, consider an MLP model with 50 neurons at some layer. If we feed a
  MNIST image (784-dimensional vector) to the MLP, outputs of this layer is a 50-dimensional vector that may be treated
  as an embedding for this image. The better the model is trained, the better the embeddings represent their original
  objects. Note: for MNIST, last classification layer provides a 10-dimensional embedding vector.

  Embeddings have a relatively low number of dimensions, but not low enough to be adequately visualized. It is suggested
  to use t-SNE algorithm to further decrease number of dimensions to 3 or 2.

  You are to prepare a code for visualizing embeddings for MNIST (or other dataset) images, e.g.
  like [here](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding#/media/File:T-SNE_Embedding_of_MNIST.png)
  . Similarly to the task above, you should perform visualization experiments, draw some conclusions and present the
  results. You can try applying t-SNE to raw image vectors, MLP activations (pre-ReLU preferably) for different layers
  and network structures, hyperparameters, etc., and compare them by some clustering metric or just what looks "better".
- **Adversarial images generation**<br/>
  Adversarial image is an image crafted in a special way which purpose is to "fool" machine learning models, e.g. to
  cause a model to missclassify this image. The algorithm for generating such images usually starts with a regular image
  that is classified correctly by the model and then modifies it slightly so that it is still perceived the same by
  human, but differently by the model, leading to missclassification.

  The task is to implement algorithm for generating adversarial examples based on the MLP classifier. Generation is to
  be performed for images from the dataset of your choice (from previous task) using targeted and untargeted FGM-like
  methods
  ([link1](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm),
  [link2](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)). You might present examples for images across
  different classes, methods, hyperparameters, architectures etc. Key metrics are original confidence, confidence for
  adversarial image, their difference magnitude. Besides, would be interesting to see targeted transitions from one
  class to another, for example transition from MNIST digit 4 to 9.
- **Image gradients and first layer weights visualization**<br/>
  Gradient of per-class predictions over the input image can be useful in understanding what the model "looks" for when
  performing classification. Large magnitude of a gradient values reflects the importance of corresponding pixel for
  classification, and its sign (positive or negative) reflects whether the pixel should be brighter of dimmer for
  classification confidence to increase.

  Another visualization that might be useful is the visualization of the first MLP layer's weights. As each neuron in
  the first layer is connected to each pixel of an input image we can use values of this neuron's weights to see which
  pixels will strongly contribute to neuron's activation. The magnitude of the weight reflects the "strength" of the
  contribution and its sign reflects whether the brightness of the pixel leads to increase or decrease of the
  activation.

  In both cases you can reshape the visualized value to the shape of a dataset image and plot it as a grayscale mask.

  For the case of the gradient visualization it would be interesting to demonstrate (1) gradient of ground truth class
  prediction over an image, (2) gradient of non ground truth class predictions over an image. Ofcourse you should also
  perform these experiments for images from different classes.

  For the case of weight visualization you may plot weight masks for individual neurons or compute the mean of weights
  corresponding to the same pixel across all neurons of the first layer. In the first case it could be beneficial to
  reduce the number of neurons in the first layer because the visualization should become more meaningful as the small
  number of neurons makes each of them more distinct. You should also try to visualize weights for the case when
  MLP has no hidden layer, i.e. it has a single linear layer transforming image pixels straight into classification
  logits.
- **Training with image augmentation**<br/>
  Modern computer vision machine learning models are often trained with augmentation techniques. Image augmentation is
  used to enrich limited training data by introducing slight random alterations such as addition of small noise,
  rotations, shifting and other.

  You should implement some popular image augmentation methods for images from your dataset, train on images with
  augmentation and perform experiments showing how such training improves the performance of the MLP model. You may vary
  strength of augmentation, sets of techniques employed.
- **Training for edge detection task**<br/>
  During the course we have considered the classification machine learning problem. The more general computer vision
  task is semantic segmentation, where the task is to predict a category for each pixel of an image. Since the datasets
  we consider do not have that kind of annotation, you are to generate artificial semantic annotations yourselves. It is
  suggested to generate annotations for edge detection task since it is rather simple using existing computer vision
  algorithms such as Canny's edge detection algorithm. This should apply to MNIST-like datasets only, since they have
  bright objects over black background.

  First, you are to pre-compute edge binary masks for images from the dataset using
  Canny's [algorithm](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html). Then you will need to modify the MLP
  architecture to output the same number of sigmoid predictions as the number of input pixels (i.e. a prediction per
  pixel) to perform per-pixel binary classification. Next, you should apply binary cross entropy loss function for each
  output pixel against pre-computed edge pixels. Another option is to implement
  [Mean Squared Error](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) loss used for regression tasks.
  The total loss for each image would be the mean per pixel loss. The metric for measuring the performance of the model
  might be Precision, Recall, F1-score or Intersection over Union (IoU).
- **Transfer learning**<br/>
  Transfer learning is a popular technique in machine learning in general and in computer vision in particular. In the
  most basic sense it is when one takes a model pre-trained on some large dataset and uses it to solve other downstream
  task of their need. In such case the main part of the network referred to as backbone is taken as is but the last
  layer, e.g. a classification head, is replaced by the new one used specifically for solving the downstream task. The
  modified network is then fine-tuned on the downstream task, however the backbone weights are frozen and only the new
  head layer is let to be trained. The effect of such approach is twofold: firstly, as the backbone is well-trained it
  serves as great feature extractor, secondly, since the number of parameters in the new detection (or some other)
  head is small, it is relatively easy to train it and obtain good results.

  Despite the fact that the MLP architecture considered in our case cannot be considered as a strong backbone, we can
  still showcase the core principles of this approach. First, you are to train one MLP network that will serve as
  backbone. Then you are to replace the last layer of this MLP by another one that will be used to solve a
  classification task. It is suggested to train backbone for auto-encoding images from your dataset. The auto-encoder
  architecture will have two fully connected layers. It takes an image, transforms it to a significantly smaller number
  of dimensions (bottleneck) and then tries to reconstruct the original image for example by per-pixel binary cross
  entropy. Because the number of neurons in the bottleneck is small, we can expect it to be a good feature extractor.
  After the auto-encoder is trained, the layer after the bottleneck is thrown out and replaced by the new fully
  connected layer that will perform classification. When training for classification, we freeze the parameters of the
  previously trained layer and let only the classification layer to be trained.

  When the described pipeline is ready, you may perform various experiments, for example by changing the size of the
  bottleneck, number of epochs that the auto-encoder or final classificator is trained, etc. It is also important to
  compare the accuracy of such approach to the case when we train for classification task from the start.
- **Training on unbalanced dataset**<br/>
  The datasets considered in this course are balanced in the sense that there are equal number of samples of each
  category. Usually that is not the case which worsens the performance of the trained model. Hence, some efforts have to
  be applied to deal with this issue. The most popular approaches to tackle the imbalance problem are (1) oversampling
  of examples from infrequent classes, (2) per class loss weighting, (3) special loss functions, such as Focal loss.

  Since both our 2D dataset and the simple image dataset you have chosen are balanced, you will need to introduce
  artificial imbalance yourselves as if it was present in the original dataset. Then you are to deal with this problem
  using the aforementioned approaches. You should compare the accuracy of vanilla training with the modified training
  scheme using metrics such as mean per-class accuracy, precision, recall and F1-score. In the case of 2D binary
  classification dataset you should increase the proportion of negative class (target=0). For the multiclass problem you
  are free to test different imbalance scenarios, e.g. single overwhelming class, multiple major classes, etc.
  