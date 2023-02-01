from abc import ABC, abstractmethod

import numpy as np
from mnist import MNIST
from skimage.util.shape import view_as_windows
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder


class Layer(ABC):
    @abstractmethod
    def get_output_dimension(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dl_dy, lr):
        pass


class Conv(Layer):
    def __init__(self, input_dimensions, filter_count, kernel_size, stride, padding):
        self.input_dimensions = input_dimensions
        self.filter_count = filter_count
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.filter_count, self.input_dimensions[0], kernel_size, kernel_size)  # / 10
        self.weights *= np.sqrt(2 / np.prod([self.input_dimensions]))
        d, w, h = self.get_output_dimension()
        self.bias = np.random.randn(d, w, h)

        # self.weights = np.full((self.filter_count, self.input_dimensions[0], kernel_size, kernel_size), .5) / 100
        # # self.weights *= np.sqrt(2) / np.prod([self.weights.shape])
        # self.bias = np.full(self.get_output_dimension(), .1)

        self.x = None

    def get_output_dimension(self):
        # W2 = (W1 − F + 2P) / S + 1
        w = int(np.floor((self.input_dimensions[1] - self.kernel_size + 2 * self.padding) / self.stride + 1))
        h = int(np.floor((self.input_dimensions[2] - self.kernel_size + 2 * self.padding) / self.stride + 1))
        d = self.filter_count
        return d, w, h

    @staticmethod
    def correlate3d(image, weights, bias, stride):
        output = []

        channel_windows = np.array(
            [view_as_windows(image[i], (weights.shape[2], weights.shape[3]), step=stride) for i in
             range(image.shape[0])])

        for i in range(weights.shape[0]):
            output.append(
                np.sum([np.tensordot(channel_windows[j], weights[i][j], axes=((2, 3), (0, 1)))
                        for j in range(channel_windows.shape[0])], axis=0)
                + bias[i])

        return np.array(output)

    def forward(self, x):
        # x = np.array([np.pad(x[i], pad_width=self.padding) for i in range(x.shape[0])])
        n_pad = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        self.x = np.pad(x, pad_width=n_pad, mode='constant', constant_values=0)

        output = np.array(
            [self.correlate3d(self.x[m], self.weights, self.bias, self.stride) for m in range(self.x.shape[0])])

        return output

    def backward(self, dl_dy, lr):
        input_depth = self.input_dimensions[0]

        dl_dw = np.zeros(self.weights.shape)
        for i in range(self.filter_count):
            for j in range(input_depth):
                for m in range(dl_dy.shape[0]):
                    # m_weights = np.array([np.take(dl_dy, indices=i, axis=1)])
                    # m_images = np.take(self.x, indices=j, axis=1)
                    # dl_dw[i][j] = self.correlate3d(image=m_images, weights=m_weights, bias=[0], stride=self.stride)
                    dl_dw[i][j] += self.correlate3d(image=np.array([self.x[m][j]]), weights=np.array([[dl_dy[m][i]]]),
                                                    bias=[0],
                                                    stride=self.stride)[0]

        dl_db = np.sum(dl_dy, axis=0)

        dim = self.input_dimensions
        dl_dx = np.zeros(tuple([self.x.shape[0]]) + dim)
        dl_dx_padded = np.zeros((dim[0], dim[1] + 2 * self.padding, dim[2] + 2 * self.padding))
        d, w, h = self.get_output_dimension()
        for m in range(dl_dy.shape[0]):
            for k in range(d):
                for i in range(h):
                    for j in range(w):
                        h_start = i * self.stride
                        h_stop = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_stop = w_start + self.kernel_size

                        dl_dx_padded[:, h_start:h_stop, w_start:w_stop] += self.weights[k, :, :, :] * dl_dy[m, k, i, j]
            if self.padding == 0:
                dl_dx[m] = dl_dx_padded
            else:
                dl_dx[m] = dl_dx_padded[:, self.padding:-self.padding, self.padding:-self.padding]

        self.weights = self.weights - lr * dl_dw  # Updating weights
        self.bias = self.bias - lr * dl_db  # Updating bias

        return dl_dx


class ReLU(Layer):
    def __init__(self, input_dimensions):
        self.input_dimensions = input_dimensions
        self.x = None

    def get_output_dimension(self):
        return self.input_dimensions

    def forward(self, x):
        self.x = x
        return x * (x > 0)

    def backward(self, dl_dy, lr):
        dy_dx = self.x
        dy_dx[dy_dx >= 0] = 1
        dy_dx[dy_dx < 1] = 0

        dl_dx = dl_dy * dy_dx
        return dl_dx


class Pool(Layer):
    def __init__(self, input_dimensions, pool_size, stride):
        self.input_dimensions = input_dimensions
        self.pool_size = pool_size
        self.stride = stride

        self.x = None

    def get_output_dimension(self):
        w = int(np.floor((self.input_dimensions[1] - self.pool_size) / self.stride + 1))
        h = int(np.floor((self.input_dimensions[2] - self.pool_size) / self.stride + 1))
        d = self.input_dimensions[0]
        return d, w, h

    def forward(self, x):
        self.x = x

        channel_windows = np.array([[view_as_windows(x[m][i], (self.pool_size, self.pool_size), step=self.stride) for i
                                     in range(x[m].shape[0])] for m in range(x.shape[0])])

        return np.max(channel_windows, axis=(4, 5))

    def backward(self, dl_dy, lr):
        dl_dx = np.zeros(self.x.shape)
        d, w, h = self.get_output_dimension()

        for m in range(dl_dy.shape[0]):
            for k in range(d):
                for i in range(h):
                    for j in range(w):
                        h_start = i * self.stride
                        h_stop = h_start + self.pool_size
                        w_start = j * self.stride
                        w_stop = w_start + self.pool_size

                        x_slice = self.x[m, k, h_start:h_stop, w_start:w_stop]
                        mask = x_slice == x_slice.max()
                        dl_dx[m, k, h_start:h_stop, w_start:w_stop] += mask * dl_dy[m, k, i, j]

        return dl_dx


class Flatten(Layer):
    def __init__(self, input_dimensions):
        self.input_dimensions = input_dimensions

    def get_output_dimension(self):
        return np.prod(list(self.input_dimensions))

    def forward(self, x):
        return np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))

    def backward(self, dl_dy, lr):
        temp = np.reshape(dl_dy, (tuple([dl_dy.shape[0]]) + self.input_dimensions))
        return temp


class FullyConnected(Layer):
    def __init__(self, input_dimension, output_dimension):
        self.output_dimension = output_dimension

        self.weights = np.random.randn(output_dimension, input_dimension)
        self.weights *= np.sqrt(2 / input_dimension)
        self.bias = np.random.randn(output_dimension)

        # self.weights = np.random.uniform(low=-1.0, high=1.0, size=(output_dimension, input_dimension))
        # self.weights = np.full((output_dimension, input_dimension), .15) / 100
        # self.bias = np.full((output_dimension,), .20)

        self.x = None

    def get_output_dimension(self):
        return self.output_dimension

    def forward(self, x):
        self.x = x
        return (self.weights @ self.x.T).T + self.bias

    def backward(self, dl_dy, lr):
        # dl_dw = np.reshape(dl_dy, (len(dl_dy), 1)) @ np.reshape(self.x, (1, len(self.x)))
        dl_dw = dl_dy.T @ self.x
        dl_db = np.sum(dl_dy, axis=0)
        dl_dx = dl_dy @ self.weights

        self.weights = self.weights - lr * dl_dw  # updating weights
        self.bias = self.bias - lr * dl_db  # updating bias

        return dl_dx


class Softmax(Layer):
    def __init__(self, input_dimensions):
        self.input_dimensions = input_dimensions

    def get_output_dimension(self):
        return self.input_dimensions

    def forward(self, x):
        # return np.exp(x - max(x)) / np.sum(np.exp(x - max(x)))
        return np.array([np.exp(x[m] - max(x[m])) / np.sum(np.exp(x[m] - max(x[m]))) for m in range(x.shape[0])])

    def backward(self, dl_dy, lr):
        return dl_dy


class CNN:
    def __init__(self, train, validation, batch_size, learning_rate):
        self.x_train, self.y_train = train
        self.x_valid, self.y_valid = validation
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.total_samples = self.x_train.shape[0]
        self.latest_input_dimensions = self.x_train[0].shape
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        self.latest_input_dimensions = layer.get_output_dimension()

    def train(self, epoch):
        for e in range(epoch):
            print("------------ Epoch", e + 1, "------------")
            for b in range(0, self.total_samples, self.batch_size):
                x_batch = self.x_train[b:b + self.batch_size]
                y_batch = self.y_train[b:b + self.batch_size]

                # forward propagation starts
                y_hat = self.layers[0].forward(x_batch)
                for i in range(1, len(self.layers)):
                    y_hat = self.layers[i].forward(y_hat)
                # forward propagation ends

                y_hat_clipped = np.clip(y_hat.copy(), 1e-20, 1)
                # loss = -1 * np.sum(np.log(y_hat_clipped) * y_batch) / self.batch_size
                loss = np.sum(-1 * np.log(np.sum((y_hat_clipped * y_batch), axis=1))) / self.batch_size
                if b % (self.batch_size * 10) == 0:
                    print(b, "-", b + self.batch_size - 1, "CE loss", loss)

                # backward propagation starts
                layer = self.layers[len(self.layers) - 1]
                output_gradient = layer.backward(y_hat - y_batch, self.learning_rate)
                for i in reversed(range(len(self.layers) - 1)):
                    layer = self.layers[i]
                    output_gradient = layer.backward(output_gradient, self.learning_rate)
                # backward propagation ends

            self.validation()
            print()

    def validation(self):
        y_hat = self.layers[0].forward(self.x_valid)
        for i in range(1, len(self.layers)):
            y_hat = self.layers[i].forward(y_hat)

        loss = np.sum(-1 * np.log(np.sum((y_hat * self.y_valid), axis=1))) / self.x_valid.shape[0]
        y_true = np.argmax(self.y_valid, axis=1)
        y_prediction = np.argmax(y_hat, axis=1)

        print()
        print("Validation CE loss:", loss)
        print("Accuracy:", accuracy_score(y_true, y_prediction))
        print("Macro-F1:", f1_score(y_true, y_prediction, average='macro'))


if __name__ == "__main__":
    np.random.seed(1)

    print("Loading Dataset...")
    mnist = MNIST('./MNIST')
    X_train, Y_train = mnist.load_training()
    X_train = np.asarray(X_train).astype(np.float64)
    Y_train = np.asarray(Y_train).astype(np.int32)
    X_test, Y_test = mnist.load_testing()
    X_test = np.asarray(X_test).astype(np.float32)
    Y_test = np.asarray(Y_test).astype(np.int32)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_train /= 255
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))
    X_test /= 255

    Y_train = Y_train.reshape(len(Y_train), 1)
    Y_train = OneHotEncoder(sparse=False).fit_transform(Y_train)
    Y_test = Y_test.reshape(len(Y_test), 1)
    Y_test = OneHotEncoder(sparse=False).fit_transform(Y_test)

    X_valid = X_test[0:int(X_test.shape[0] / 2)]
    Y_valid = Y_test[0:int(X_test.shape[0] / 2)]
    print("Dataset Loaded.")

    model = CNN(train=(X_train[0:500], Y_train[0:500]), validation=(X_valid[0:20], Y_valid[0:20]), batch_size=10,
                learning_rate=0.01)

    model.add(Conv(input_dimensions=model.latest_input_dimensions, filter_count=6, kernel_size=5, stride=1, padding=2))
    model.add(ReLU(input_dimensions=model.latest_input_dimensions))
    model.add(Pool(input_dimensions=model.latest_input_dimensions, pool_size=2, stride=2))

    model.add(Conv(input_dimensions=model.latest_input_dimensions, filter_count=12, kernel_size=5, stride=1, padding=0))
    model.add(ReLU(input_dimensions=model.latest_input_dimensions))
    model.add(Pool(input_dimensions=model.latest_input_dimensions, pool_size=2, stride=2))

    model.add(
        Conv(input_dimensions=model.latest_input_dimensions, filter_count=100, kernel_size=5, stride=1, padding=0))
    model.add(ReLU(input_dimensions=model.latest_input_dimensions))

    model.add(Flatten(input_dimensions=model.latest_input_dimensions))

    model.add(FullyConnected(input_dimension=model.latest_input_dimensions, output_dimension=10))

    model.add(Softmax(input_dimensions=model.latest_input_dimensions))

    model.train(epoch=2)
