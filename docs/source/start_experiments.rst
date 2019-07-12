Starting the Experiments
========================

To start the experiments, you have to run the
start_experiments script, which takes two
arguments: model and dataset. You have to insert
your configuration (which is a function tagged with
'\@ex.named_config') for the model and
training parameters as the model argument.
Analogously, you have to insert the dataset configuration
(which is a function tagged with '\@ex.named_config')
as argument for dataset. In the following, the model and
the training configuration function is defined as follows
in the experiments script:

.. code-block:: python
   :emphasize-lines: 3,5

    @ex.named_config
    def random_search_renet():
        """
        Configuration for the ReNet model.
        """
        model_params = {
            'model': 'renet',
            'reNet_layer_dim': random.choice(range(1, 5)),
            'linear_layer_dim': random.choice(range(1, 5)),
            'reNet_hidden_dim': [],
            'linear_hidden_dim': [],
            'dropout_rate': [],
            'input_dim': (32, 32, 3),  # H, W, C
            'rnn_types': [],
            'window_size': [],
            'output_dim': 10
        }
        model_params['reNet_hidden_dim'] = [
            int(loguniform(math.log(100), math.log(300)))
            for _ in range(model_params['reNet_layer_dim'])]

        model_params['dropout_rate'] = [
            (int(loguniform(math.log(0.2), math.log(0.6))),
             int(loguniform(math.log(0.2), math.log(0.6))))
            for _ in range(model_params['reNet_layer_dim'])]
        model_params['rnn_types'] = [random.choice(['GRU', 'LSTM', 'RNN'])
                                     for _ in
                                     range(model_params['reNet_layer_dim'])]
        try:
            model_params['window_size'] = random.choice(
                get_partitions(model_params['input_dim'][0],
                               model_params['input_dim'][1],
                               model_params['reNet_layer_dim']))
        except IndexError:
            raise IndexError(
                "<reNet_layer_dim> can not be used "
                + "to create partitions out of <input_dim>")
        model_params['linear_hidden_dim'] = [
            int(loguniform(math.log(100), math.log(300)))
            for _ in range(model_params['linear_layer_dim'])]

        train_params = {
            'max_epochs': 200,
            'batch_size': 128,
            'early_stopping_patience': 20,
            'acc_metric': 'classification',
            'class_dim': 1,
            'loss': random.choice(['CrossEntropyLoss']),
            'optimizer': random.choice(['RMSprop', 'Adam', 'Adadelta']),
            'opt_lr': float(10 ** np.random.uniform(-3, 0))
        }

The corresponding dataset configuration function is defined as:

.. code-block:: python
   :emphasize-lines: 3,5

    @ex.named_config
    def renet_dataset():
        """
        Dataset file names for ReNet.
        """
        dataset = {
            'train_input_filename':
                '../data/CIFAR10/input_data/train_renet_data_aug',
            'train_target_filename':
                '../data/CIFAR10/input_data/train_renet_labels_aug',
            'validation_input_filename':
                '../data/CIFAR10/input_data/validation_renet_data_aug',
            'validation_target_filename':
                '../data/CIFAR10/input_data/validation_renet_labels_aug',
            'test_input_filename':
                '../data/CIFAR10/input_data/test_renet_data_aug',
            'test_target_filename':
                '../data/CIFAR10/input_data/test_renet_labels_aug'
        }

To start the experiments with those configurations,
you have to call the following command:

.. code-block:: bash
   :emphasize-lines: 3,5

   python start_experiments.py --model random_search_renet --dataset renet_dataset

Continuing Experiments after KeyboardInterrupt
-----------------------------------------------

It is possible to interrupt the experiments and then
continuing them via the continue_experiment.py script.
You just have to pass the directory and the filename
of the desired experiment as arguments like:

.. code-block:: bash
   :emphasize-lines: 3,5

   python continue_experiment.py -d saved_models -f renet_model_state_dict.pth

