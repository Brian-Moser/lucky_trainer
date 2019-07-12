Introduction
==================

The main purpose of this project is to support other PyTorch projects by creating an easy way for training models, especially for RNN models w.r.t. CV tasks.

For example, this code block below runs an experiment with GoogLeNet (implementation from PyTorch) on tiny ImageNet (with 200 classes). The train parameters are set in one dictionary (see train_params).
This documentation will explain all keywords, the dictionary below defines an experiment without train set (skip_test), which takes 60 epochs and batch size of 128.
It is minimizing the Cross-Entropy loss function. As an accuracy, top-1 and top-5 accuracy is reported, class_dim defines the dimension of the classes in the output tensor.
Every keyword starting with "opt_" is a parameter to control the arguments of the optimizer given by PyTorch (since different optimizers can have different arguments). In this case,
the optimizer Adam has a L2 regularization constant of 1e-4. In addition, a learn rate decay of 0.1 at epoch 20 and 40 is given.

The dataset is defined in the dataset_params, where only the path is needed. However, the dataset needs to be preprocessed in a special way (already a PyTorch dataset, not wrapped in an iterator like PyTorch's data loader).
The next step is to wrap this dataset with a data loader.

The most important step is then to define the model you want to train, in this case GoogLeNet.
After this, you can run the experiment with the start_training function and have fun!

.. code-block:: python
   :emphasize-lines: 3,5

    import torch.nn as nn
    from torchvision.models import GoogLeNet
    from lucky_trainer.utils import get_dataset, start_training

    # Parameter settings
    train_params = {
        'skip_test': True,
        'max_epochs': 60,
        'batch_size': 128,
        'acc_metric': 'classification_1_and_5',
        'class_dim': 1,
        'loss': 'CrossEntropyLoss',
        'optimizer': 'Adam',
        'opt_weight_decay': 1e-4,
        'lr_decay_gamma': 0.1,
        'lr_decay_step_size': [20, 40],
    }

    # Path of the datasets
    dataset_params = {
        'train_filename':
            path_file +
            '/../data/tiny_ImageNet/input_data/train_tiny_imagenet',
        'validation_filename':
            path_file +
            '/../data/tiny_ImageNet/input_data/valid_tiny_imagenet'
    }

    # Load iterable datasets
    train_loader = get_dataset(
        dataset_params['train_filename'],
        train_params['batch_size']
    )
    validation_loader = get_dataset(
        dataset_params['validation_filename'],
        train_params['batch_size'],
        shuffle=False
    )

    # Model definition (with downsizing amount of classes from 1000 to 200
    model = GoogLeNet(aux_logits=False)
    model.fc = nn.Linear(in_features=1024, out_features=200)

    # Train the model
    start_training(model, {}, train_params, dataset_params,
                   train_loader, validation_loader, None,
                   output_directory, 'Some_GoogLeNet_Experiment')