from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import struct


class MuvrSequential(Sequential):
    def save_weights(self, filename):
        weights = MuvrSequential.deep_flatten(self.get_weights())
        weights_string = MuvrSequential.model2string(weights)
        with open(filename, 'wb') as f:
            f.write(weights_string)

    def save_layers(self, filename):
        with open(filename, 'w') as f:
            f.writelines(self.layers_as_string)

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
