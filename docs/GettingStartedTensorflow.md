# Getting Started with Tensorflow 1.1 in DIGITS

Table of Contents
=================
* [Enabling Support For Tensorflow In DIGITS](#enabling-support-for-tensorflow-in-digits)
* [Selecting Tensorflow When Creating A Model In DIGITS](#selecting-tensorflow-when-creating-a-model-in-digits)
* [Defining A Tensorflow Model In DIGITS](#defining-a-tensorflow-model-in-digits)
    * [Provided Properties](#provided-properties)
    * [Internal Properties](#internal-properties)
    * [Tensors](#tensors)
* [Other Tensorflow Tools in DIGITS](#other-tensorflow-tools-in-digits)
    * [Provided Helpful Functions](#provided-helpful-functions)
    * [Visualization With TensorBoard](visualization-with-tensorboard)
* [Examples](#examples)
    * [Simple Auto-Encoder Network](#simple-auto-encoder-network)
    * [Specifying Specific Variables to Train](#specifying-specific-variables-to-train)
    * [Multi-GPU Training](#multi-gpu-training)
* [Tutorials](#tutorials)

## Enabling Support For Tensorflow In DIGITS

DIGITS will automatically enable support for Tensorflow if it detects that Tensorflow-gpu is installed in the system. This is done by a line of python code that attempts to ```import tensorflow``` to see if it actually imports.

If DIGITS cannot enable tensorflow, a message will be printed in the console saying: ```Tensorflow support is disabled```

## Selecting Tensorflow When Creating A Model In DIGITS

Click on the "Tensorflow" tab on the model creation page

{insert image here}

## Defining A Tensorflow Model In DIGITS

To define a Tensorflow model in DIGITS, you need to write a python class that follows this basic template

```python
class UserModel(Tower):
* 
    @model_propertyOther Tensorflow Tools in DIGITS
    def inference(self):
        # Your code here
        return model

    @model_property
    def loss(self):
        # Your code here
        return loss
```

For example, this is what it looks like for LeNet:

```python
class UserModel(Tower):

    @model_property
    def inference(self):
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        # scale (divide by MNIST std)
        x = x * 0.0125
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005) ):
            model = slim.conv2d(x, 20, [5, 5], padding='VALID', scope='conv1')
            model = slim.max_pool2d(model, [2, 2], padding='VALID', scope='pool1')
            model = slim.conv2d(model, 50, [5, 5], padding='VALID', scope='conv2')
            model = slim.max_pool2d(model, [2, 2], padding='VALID', scope='pool2')
            model = slim.flatten(model)
            model = slim.fully_connected(model, 500, scope='fc1')
            model = slim.dropout(model, 0.5, is_training=self.is_training, scope='do1')
            model = slim.fully_connected(model, self.nclasses, activation_fn=None, scope='fc2')
            return model

    @model_property
    def loss(self):
        loss = digits.classification_loss(self.inference, self.y)
        accuracy = digits.classification_accuracy(self.inference, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
```

The properties ```inference``` and ```loss``` must be defined and the class must be called ```UserModel``` and it must inherit ```Tower```. This is how DIGITS will interact with the python code.

### Provided Properties

Properties that are accessible through ```self```

Property name | Type      | Description
--------------|-----------|------------
nclasses      | number    | Number of classes (for classification datasets). For other type of datasets, this is undefined
input_shape   | Tensor    | Shape (1D Tensor) of the first input Tensor. For image data, this is set to height, width, and channels accessible by [0], [1], and [2] respectively.
is_training   | boolean   | Whether this is a training graph
is_inference  | boolean   | Whether this graph is created for inference/testing
x             | Tensor    | The input node, with the shape of [N, H, W, C]
y             | Tensor    | The label, [N] for scalar labels, [N, H, W, C] otherwise. Defined only if self.is_training is True

### Internal Properties

These properties are in the ```UserModel``` class written by the user

Property name | Return Type | Description
--------------|-------------|------------
__init()__    | None        | The constructor for the ```UserModel``` class
inference()   | Tensor      | Called during training and inference
loss()        | Tensor      | Called during training to determine the loss and variables to train

### Tensors

The network are fed with Tensorflow Tensor objects that are in [N, H, W, C] format.

## Other Tensorflow Tools in DIGITS

DIGITS provides a few useful tools to help with your development with Tensorflow.

### Provided Helpful Functions

{To be filled}

### Visualization With TensorBoard

{insert image of tensorboard here}

TensorBoard is a visualization tools provided by Tensorflow to see the graph of your neural network. DIGITS provides easy access to TensorBoard for your network while creating it. The TensorBoard can be accessed by clicking on the ```Visualize``` button under ```Custom Network``` as seen in the image below.

{insert image here}

If there is something wrong with the network model, DIGITS will automatically provide with you the stacktrace and the error message to help you understand where the problem is.

To know more about how TensorBoard works, its official documentation is availabile in the [official tensorflow documentaton](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

## Examples

### Simple Auto-Encoder Network

The following network is a simple auto encoder to demostate the structure of how to use tensorflow in DIGITS
```python
class UserModel(Tower):

    @model_property
    def inference(self):

        # the order for input shape is [0] -> H, [1] -> W, [2] -> C
        # this is because tensorflow's default order is NHWC
        model = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        image_dim = self.input_shape[0] * self.input_shape[1]

        with slim.arg_scope([slim.fully_connected], 
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

            # first we reshape the images to something
            model = tf.reshape(_x, shape=[-1, image_dim])

            # encode the image
            model = slim.fully_connected(model, 300, scope='fc1')
            model = slim.fully_connected(model, 50, scope='fc2')

            # decode the image
            model = slim.fully_connected(model, 300, scope='fc3')
            model = slim.fully_connected(model, image_dim, activation_fn=None, scope='fc4')

            # form it back to the original
            model = tf.reshape(model, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
            
            return model

    @model_property
    def loss(self):

        # In an autoencoder, we compare the encoded and then decoded image with the original
        original = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        # self.inference is called to get the processed image
        model = self.inference
        loss = digits.mse_loss(original, model)

        return loss
```

### Specifying Specific Variables to Train

The following is a demonstration of how to specifying which weights we would like to use for training. This is applicable for fine tuning a model.
```python
class UserModel(Tower):

    @model_property
    def inference(self):

        model = construct_model()
        """code to construct the network omitted"""

        self.weights = {
            'weight1': tf.get_variable('weight1', [5, 5, self.input_shape[2], 20], initializer=tf.contrib.layers.xavier_initializer()),
            'weight2': tf.get_variable('weight2', [5, 5, 20, 50], initializer=tf.contrib.layers.xavier_initializer())
        }

        self.biases = {
            'bias1': tf.get_variable('bias1', [20], initializer=tf.constant_initializer(0.0)),
            'bias2': tf.get_variable('bias2', [50], initializer=tf.constant_initializer(0.0))
        }

        return model

    @model_property
    def loss(self):
        loss = calculate_loss()
        """code to calculate loss omitted"""

        # We would only like to train the variables for the 2nd layer of the network
        # as indicated by the number 2 in their suffix
        # When we specify which variables to train, all the other variables will be frozen
        return [{'loss': loss, 'vars': [self.weights['weight2'], self.biases['bias2']]}]
```

### Multi-GPU Training

<WIP>

## Tutorials

