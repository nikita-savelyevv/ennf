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
learning, perform some experiments and present your findings to the group. The more valuable are your findings, the
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
- **Training with image augmentation**<br/>
- **Training for edge detection task**<br/>
- **Transfer learning**<br/>
- **Knowledge distillation**<br/>
- **Training on unbalanced dataset**<br/>
  