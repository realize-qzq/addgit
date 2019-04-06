# Import TensorFlow and TensorFlow Eager
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Import function to generate toy classication problem
from sklearn.datasets import make_moons

# Import library for plots


# Enable eager mode. Once activated it cannot be reversed! Run just once.
tfe.enable_eager_execution()

# Generate toy dataset for classification
# X is a matrix of n_samples x n_features and represents the input features
# y is a vector with length n_samples and represents our targets
X, y = make_moons(n_samples=100, noise=0.1, random_state=2018)


class simple_nn(tf.keras.Model):
    def __init__(self):
        super(simple_nn, self).__init__()
        """ Define here the layers used during the forward-pass
            of the neural network.
        """
        # Hidden layer.
        self.dense_layer = tf.layers.Dense(10, activation=tf.nn.relu)
        # Output layer. No activation.
        self.output_layer = tf.layers.Dense(2, activation=None)

    def predict(self, input_data):
        """ Runs a forward-pass through the network.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:
                logits: unnormalized predictions.
        """
        hidden_activations = self.dense_layer(input_data)
        logits = self.output_layer(hidden_activations)
        return logits

    def loss_fn(self, input_data, target):
        """ Defines the loss function used during
            training.
        """
        logits = self.predict(input_data)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=target, logits=logits)
        return loss

    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit(self, input_data, target, optimizer, num_epochs=500, verbose=50):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs.
        """
        for i in range(num_epochs):
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i == 0) | ((i + 1) % verbose == 0):
                print('Loss at epoch %d: %f' % (i + 1, self.loss_fn(input_data, target).numpy()))


X_tensor = tf.constant(X)
y_tensor = tf.constant(y)

optimizer = tf.train.GradientDescentOptimizer(5e-1)
model = simple_nn()
model.fit(X_tensor, y_tensor, optimizer, num_epochs=500, verbose=50)
