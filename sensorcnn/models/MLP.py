from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import struct


class MLP(object):
    def __init__(self, config, name):
        """
        layers example: "150 id 64 Tanh_0 32 Tanh_1 6 Tanh_2"
        """
        self.config = config
        self.name = name
        self.model = Sequential()
        layers = MLP.parse_layers(config)
        for layer in layers:
            self.add_layer(layer)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
        self.fit = self.model.fit
        self.evaluate = self.model.evaluate
        self.predict = self.model.predict

    def add_layer(self, layer):
        input_dim, output_dim, activation = layer
        self.model.add(Dense(input_dim=input_dim, output_dim=output_dim, activation=activation, init='uniform'))

    def save_weights(self, filename):
        weights = MLP.deep_flatten(self.model.get_weights())
        weights_string = MLP.model2string(weights)
        with open(filename, 'wb') as f:
            f.write(weights_string)

    def save_layers(self, filename):
        with open(filename, 'w') as f:
            f.writelines(self.config)

    @staticmethod
    def parse_layers(config):
        xs = config.split()
        dims = [int(x) for i, x in enumerate(xs) if i % 2 == 0]
        activations = [x.lower().split("_")[0] for i, x in enumerate(xs[2:]) if i % 2 == 1]
        return [x for x in zip(dims[:-1], dims[1:], activations)]

    @staticmethod
    def deep_flatten(xs):
        result = []
        for x in xs:
            if len(x.shape) == 1:
                result.extend(x)

            else:
                i, j = x.shape
                result.extend(x.reshape((i * j,)))

        return result

    @staticmethod
    def model2string(weights):
        """Serialize the model at the path to a string representation."""
        return struct.pack('f' * len(weights), *weights)
