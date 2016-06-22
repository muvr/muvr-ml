import numpy as np
import pandas as pd
from functools import reduce
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from models import MuvrSequential
from dataset.utils import *
from train.utils import *


def test_mlp_1():
  model = MuvrSequential()
  #TODO: pass the name to the constructor
  model.name = "mlp_1"
  #TODO: infer the layers as string from the netowrk config
  model.layers_as_string = "750 id 150 tanh 100 tanh 4 tanh"
  model.add(Dense(input_dim=750, output_dim=150, activation="tanh", init='uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=150, output_dim=100, activation="tanh", init='uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=100, output_dim=4, activation="tanh", init='uniform'))
  train_mlp(model)


def test_mlp_2():
  model = MuvrSequential()
  model.name = "mlp_2"
  model.layers_as_string = "750 id 4 tanh"
  model.add(Dense(input_dim=750, output_dim=4, activation="tanh", init='uniform'))
  train_mlp(model, nb_epoch=100)


def test_mlp1_steve():
  model = MuvrSequential()
  model.name = "mlp_1_steve"
  model.layers_as_string = "750 id 150 tanh 100 tanh 2 tanh"
  model.add(Dense(input_dim=750, output_dim=150, activation="tanh", init='uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=150, output_dim=100, activation="tanh", init='uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=100, output_dim=2, activation="tanh", init='uniform'))
  setup_labels = ["setup_resistanceTargeted:arms/dumbbell-biceps-curl", "setup_resistanceTargeted:arms/triceps-extension"]
  train_mlp(model, nb_epoch=100, data_path="/data/Setup Data/Steve/", setup_labels=setup_labels)


def test_mlp2_steve():
  model = MuvrSequential()
  model.name = "mlp_2_steve"
  model.layers_as_string = "750 id 2 tanh"
  model.add(Dense(input_dim=750, output_dim=2, activation="tanh", init='uniform'))
  setup_labels = ["setup_resistanceTargeted:arms/dumbbell-biceps-curl", "setup_resistanceTargeted:arms/triceps-extension"]
  train_mlp(model, nb_epoch=100, data_path="/data/Setup Data/Steve/", setup_labels=setup_labels)


def train_mlp(model, optimizer=None, nb_epoch=30, data_path="/data/Setup Data/", setup_labels=None):
  all_samples = {}
  for filename in csv_file_iterator(data_path):
      samples = load_from_csv(filename)
      all_samples[filename] = samples

  from functools import reduce
  df = reduce(lambda x, y: x.append(y), all_samples.values())
  if not setup_labels:
    setup_labels = [l for l in df['label'].value_counts().keys() if l.startswith("setup_")]

  setup_labels_map = {l:i+1 for (i,l) in enumerate(setup_labels)}
  label_to_idx = lambda l: setup_labels_map.get(l, 0)

  df['target'] = df['label'].map(label_to_idx)
  positive_targets = df[df['target'] > 0]
  targets = positive_targets[['target']].as_matrix()
  samples = df[['x', 'y', 'z']].as_matrix()[positive_targets.index]

  X, y = augment_examples(samples, targets, new_sample_size=50*5)

  output_dim = len(setup_labels)
  Y = np.zeros((y.shape[0], output_dim))
  iy = np.array([[i, yi-1] for (i, yi) in enumerate(y) if yi > 0])
  Y[iy[:, 0], iy[:, 1]] = 1

  # Split Data
  X_train, X_test, Y_train, Y_test = split(X, Y)

  if not optimizer:
    optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["accuracy"])

  model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=128, verbose=0)

  # Printing the results
  train_score = model.evaluate(X_train, Y_train, batch_size=128, verbose=0)
  test_score = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
  print("\ntrain score\tcost: %s\taccuracy: %s" % tuple(train_score))
  print("test score\tcost: %s\taccuracy: %s" % tuple(test_score))

  # Export result into csv file
  y_true = np.array([np.argmax(yi) + 1 if np.sum(yi) > 0 else 0 for yi in Y_test])
  y_pred = np.array([np.argmax(yi) + 1 if np.sum(yi) > 0 else 0 for yi in model.predict(X_test)])
  evaluation_tabel = evaluate(y_true, y_pred, setup_labels)

  # Continue testing on the test set and export the model
  model.fit(X_test, Y_test, nb_epoch=nb_epoch, batch_size=128, verbose=0)
  labels = [l.split("setup_")[-1] for l in setup_labels]
  export(model, evaluation_tabel, labels)
