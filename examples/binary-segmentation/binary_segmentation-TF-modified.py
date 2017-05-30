class UserModel(Tower):
    
    @model_property
    def inference(self):
        _x = tf.reshape(self.x, shape=[-1, 1, 32, 32])
        _x = tf.transpose(_x, [0, 2, 3, 1])
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],  
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005) ):
    
            model = slim.conv2d(_x, 32, [3, 3], padding='SAME', scope='conv1') # 1*H*W -> 32*H*W
            model = slim.conv2d(model, 1024, [16, 16], padding='VALID', scope='conv2', stride=16) # 32*H*W -> 1024*H/16*W/16
            model = slim.conv2d_transpose(model, 1, [16, 16], stride=16, padding='VALID', activation_fn=None, scope='deconv_1')
            model = out = tf.transpose(model, [0, 3, 1, 2])
            return model
    
    @model_property
    def loss(self):
        y = tf.reshape(self.x, shape=[-1, 1, 32, 32])
        return digits.mse_loss(self.inference, y)
