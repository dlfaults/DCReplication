operator_name_dict = {'change_label': 'TCL',
                      'delete_training_data': 'TRD',
                      'unbalance_train_data': 'TUD',
                      'add_noise': 'TAN',
                      'make_output_classes_overlap': 'TCO',
                      'change_batch_size': 'HBS',
                      'change_learning_rate': 'HLR',
                      'change_epochs': 'HNE',
                      'disable_batching': 'HDB',
                      'change_activation_function': 'ACH',
                      'remove_activation_function': 'ARM',
                      'add_activation_function': 'AAL',
                      'add_weights_regularisation': 'RAW',
                      'change_weights_regularisation': 'RCW',
                      'remove_weights_regularisation': 'RRW',
                      'change_dropout_rate': 'RCD',
                      'change_patience': 'RCP',
                      'change_weights_initialisation': 'WCI',
                      'add_bias': 'WAB',
                      'remove_bias': 'WRB',
                      'change_loss_function': 'LCH',
                      'change_optimisation_function': 'OCH',
                      'change_gradient_clip': 'OCG',
                      'remove_validation_set': 'VRM'}


subject_params = {'mnist': {'epochs': 12, 'lower_lr': 0.001, 'upper_lr': 1},
                  'movie_recomm': {'epochs': 5, 'lower_lr': 0.0001, 'upper_lr': 0.001},
                  'audio': {'epochs': 50, 'lower_lr': 0.0001, 'upper_lr': 0.001, 'patience': 10},
                  'lenet': {'epochs': 50, 'lower_lr':  0.001, 'upper_lr': 0.01},
                  'udacity': {'epochs': 50, 'lower_lr':  0.00001, 'upper_lr': 0.0001}
                  }

subject_short_name = {'mnist': 'MN', 'movie_recomm': 'MR', 'audio': 'SR', 'lenet': 'UE', 'udacity': 'UD'}
