import sys
from functools import reduce
import numpy as np
import pandas as pd

from models import MuvrSequential
from dataset.utils import *
from train.utils import *

def test_mlp():
  all_samples = {}
  for filename in csv_file_iterator("/data/Setup Data/Mo"):
      samples = load_from_csv(filename)
      all_samples[filename] = samples

  from functools import reduce
  df = reduce(lambda x, y: x.append(y), all_samples.values())

  setup_labels = [l for l in df['label'].value_counts().keys() if l.startswith("setup_")]
  setup_labels_map = {l:i+1 for (i,l) in enumerate(setup_labels)}
  label_to_idx = lambda l: setup_labels_map.get(l, 0)

  df['target'] = df['label'].map(label_to_idx)

  positive_targets = df[df['target'] > 0]

  targets = positive_targets[['target']].as_matrix()
  samples = df[['x', 'y', 'z']].as_matrix()[positive_targets.index]

  print("Shapes before augmentation")
  print(samples.shape)
  print(targets.shape)

  X, y = augment_examples(samples, targets, new_sample_size=50*5)

  print("Shapes after augmentation")
  print(X.shape)
  print(y.shape)

  Y = np.zeros((y.shape[0], 4))
  iy = np.array([[i, yi-1] for (i, yi) in enumerate(y) if yi > 0])
  Y[iy[:, 0], iy[:, 1]] = 1

  # Split Data
  X_train, X_test, Y_train, Y_test = split(X, Y)

  # Training
  from keras.models import Sequential
  from keras.layers.core import Dense, Dropout, Activation
  from keras.optimizers import SGD

  model = MuvrSequential()
  model.name = "meshmesh"
  model.add(Dense(input_dim=X.shape[1], output_dim=150, activation="tanh", init='uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=150, output_dim=100, activation="tanh", init='uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=100, output_dim=4, activation="tanh", init='uniform'))

  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

  model.fit(X_train, Y_train, nb_epoch=30, batch_size=128)

  # Printing the results
  train_score = model.evaluate(X_train, Y_train, batch_size=128)
  test_score = model.evaluate(X_test, Y_test, batch_size=128)
  print(train_score)
  print(test_score)

  # Export result into csv file
  y_true = np.array([np.argmax(yi) + 1 if np.sum(yi) > 0 else 0 for yi in Y_test])
  y_pred = np.array([np.argmax(yi) + 1 if np.sum(yi) > 0 else 0 for yi in model.predict(X_test)])
  evaluation_tabel = evaluate(y_true, y_pred, setup_labels)

  # Continue testing on the test set and export the model
  model.fit(X_test, Y_test, nb_epoch=30, batch_size=128)
  labels = [l.split("setup_")[-1] for l in setup_labels]
  export(model, evaluation_tabel, labels)
