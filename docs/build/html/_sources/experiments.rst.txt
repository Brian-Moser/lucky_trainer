Experiments Setup
==================

The main contribution of this project is the easy way of
finding the right recurrent CV model for a given tasks
via random search. This is mainly realized by the
experiments setup. You have to setup two functions in the
experiments script, tagged with '\@ex.named_config'. One
function defines the parameters and one the dataset location.
The dataset should be defined like the following:

.. code-block:: python
   :emphasize-lines: 3,5

   @ex.named_config
   def bidi_lstm_datasets_row():    # Set new function name
       """
       Dataset file names for BidiLSTM (row-wise).
       """
       dataset = {
           # Keep the key names, change the values '..data/....'
           'train_input_filename':
               '../data/CIFAR10/input_data/train_row_data',
           'train_target_filename':
               '../data/CIFAR10/input_data/train_row_labels',
           'validation_input_filename':
               '../data/CIFAR10/input_data/validation_row_data',
           'validation_target_filename':
               '../data/CIFAR10/input_data/validation_row_labels',
           'test_input_filename':
               '../data/CIFAR10/input_data/test_row_data',
           'test_target_filename':
               '../data/CIFAR10/input_data/test_row_labels'
       }

According to the parameters, you also have to define a function
with the same tag '\@ex.named_config' and inside of the function
you need to configure two parameter dictionaries for the experiments,
which are the parameters for the model and the parameters
for the training itself (for example batch size). In the end,
you have something like:

.. code-block:: python
   :emphasize-lines: 3,5

   @ex.named_config
   def random_search_bidi_lstm():
       """
       Configuration for the BidiLSTM model.
       """
       model_params = {
           'model': 'bidi_lstm',
           'layer_dim': random.choice(range(1, 5)),
           'hidden_dim': [],
           'output_dim': 10,
           'dropout_rate': []
       }
       model_params['hidden_dim'] = [int(loguniform(math.log(100), math.log(300)))
                                     for _ in range(model_params['layer_dim'])]
       model_params['dropout_rate'] = [
           float(loguniform(math.log(0.25), math.log(0.75)))
           for _ in range(model_params['layer_dim'])]

       train_params = {
           'max_epochs': 200,
           'batch_size': 128,
           'early_stopping_patience': 10,
           'acc_metric': 'classification',
           'class_dim': 1,
           'loss': random.choice(['CrossEntropyLoss']),
           'optimizer': random.choice(['RMSprop', 'Adam', 'Adadelta']),
           'opt_lr': float(10 ** np.random.uniform(-3, 0))
       }

The detailed description of the parameters will be explained in
the following.

Model Parameters
----------------

The model parameters differ for each model because of
the architecture. Thus, you have to configure the model
parameters individually. In the following, this
documentation guides you through each model and tells
you the particularities.


Bidi-Lstm
~~~~~~~~~

For Bidi-LSTM, you only have 4 parameters. The 'layer_dim'
defines how many layers are used. The 'output_dim' defines
the amount of output units. Then you have define the amount
of units for each layer within the 'hidden_dim' list and
you have to define the dropout rate after each layer within
the 'dropout_rate' list. You have seen above an example
for a random search configuration, the next code snippet
shows you a fixed configuration:

.. code-block:: python
   :emphasize-lines: 3,5

   model_params = {
       'layer_dim': 3,
       'hidden_dim': [100, 100, 100],
       'output_dim': 10,
       'dropout_rate': [0.5, 0.5, 0.5]
   }

ReNet
~~~~~

The definition of the parameters are pretty similar.
You have two define the dropout rate for both directions
for each ReNet Layer, which is why you have tuples instead
of scalar values. Additionally, you have to define the window
size ('window_size') and the RNN-type ('rnn_types'),
e.g. GRU or LSTM. In addition, ReNet uses FC-Layers, which
is why you also have to define the amount of linear layers
('linear_layer_dim') and the amount of units in each layer
within the list 'linear_hidden_dim'.
The next code snippet shows you a configuration for CIFAR10,
extracted from the ReNet paper:

.. code-block:: python
   :emphasize-lines: 3,5

   model_params = {
       'reNet_layer_dim': 3,
       'linear_layer_dim': 1,
       'reNet_hidden_dim': [320, 320, 320],
       'linear_hidden_dim': [4096],
       'dropout_rate': [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2)],
       'window_size': [2, 2, 2],
       'rnn_types': ["GRU", "GRU", "GRU"],
       'output_dim': 10
   }


ConvLSTM
~~~~~~~~

Again, you have to define the amount of layers
('conv_layer_dim') and the amount of units in it
('conv_hidden_dim'), which also defines the amount
of output channels. The 'patch_size' defines the
size of the patches you extract from the input
image and the 'input_kernel_size' is the
input-to-state size. The last parameter
defines the state-to-state size for each ConvLSTM
layer.
The following shows the best performing configuration
extracted from the ConvLSTM paper for Moving-MNIST.

.. code-block:: python
   :emphasize-lines: 3,5

   model_params = {
    'conv_layer_dim': 3,
    'conv_hidden_dim': [128, 64, 64],
    'patch_size': (4, 4),
    'input_kernel_size': 5,
    'kernel_size': [5, 5, 5]
   }

Training Parameters
-------------------

This section helps you to configure the training itself.
The training parameters contain at least information about
the batch size ('batch_size'), the max. amount of
epochs 'max_epochs', the loss function ('loss')
and the optimizer ('optimizer'). You can also enable
Early Stopping by just setting the 'early_stopping_patience'.

.. code-block:: python
   :emphasize-lines: 3,5

   train_params = {
       'max_epochs': 100,
       'batch_size': 64,
       'early_stopping_patience': 20,
       'loss': 'CrossEntropyImageLoss',
       'optimizer': 'RMSprop'
   }

Setting Up the Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~

The optimizer is defined by the training parameter
dictionary 'optimizer'. You can choose any optimizer
provided by PyTorch, just use the right class name
(see documentation of PyTorch). You probably want to
configure your optimizer further more by setting
up parameters like learning rate, momentum etc.
These parameters can be also set in the training
parameter dictionary, just use the prefix '\opt_'
and the parameter names defined by PyTorch.

.. code-block:: python
   :emphasize-lines: 3,5

    train_params = {
        ...,
        'optimizer': 'Adam',
        'opt_lr': 0.0001,
        'opt_betas': (0.9, 0.999),
        'opt_weight_decay': 0
    }

Setting Up the Loss Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like for the optimizer, you just have to use
the loss function name of PyTorch. Alternatively,
this project provides additionally loss functions
like Cross-Entropy for images. They can be found
in the metrics script. Feel free to add new loss
functions for your goals, you just have to
add them as a function in the CustomLoss class.

.. code-block:: python
   :emphasize-lines: 3,5

    class CustomLoss(object):
    ...
    @staticmethod
    def YourNewLossFunction(_input, _target):
        loss = ...      # Insert your operations here
        return loss

You can find an already custom implemented loss
function for the ConvLSTM in the script:

.. code-block:: python
   :emphasize-lines: 3,5

    class CustomLoss(object):
    @staticmethod
    def CrossEntropyImageLoss(_input, _target):
        return - torch.sum(
            _target*torch.log(_input) + (1-_target)*torch.log(1-_input)
        )