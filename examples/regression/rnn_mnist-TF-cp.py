from tensorflow.contrib import rnn

class UserModel(Tower):

    @model_property
    def inference(self):
        n_hidden = 28
        n_classes = self.nclasses
        n_steps = self.input_shape[0]
        n_input = self.input_shape[1]
        
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        tf.summary.image(x.op.name, x, max_outputs=1, collections=["Training Summary"])
        x = tf.squeeze(x)

        # Define weights
        weights = {
            'w1': tf.get_variable('w1', [n_hidden, self.nclasses])
        }
        biases = {
            'b1': tf.get_variable('b1', [self.nclasses])
        }
        
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=True)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        model = tf.matmul(outputs[-1], weights['w1']) + biases['b1']
        
        return model

    @model_property
    def loss(self):
        label = self.y
        model = self.inference
        loss = digits.mse_loss(label, model)
        return loss
