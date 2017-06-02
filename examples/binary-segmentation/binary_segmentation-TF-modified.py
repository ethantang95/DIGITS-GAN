# Tensorflow Triangle binary segmentation model using TensorFlow-Slim
class UserModel(Tower):
    
    @model_property
    def inference(self):
        _x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        # assert 1 == 0, str(self.input_shape[0]) + " " + str(self.input_shape[2]) + " " + str(_x.shape)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],  
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.05) ):
    
            model = slim.conv2d(_x, 32, [3, 3], padding='SAME', scope='conv1') # 1*H*W -> 32*H*W
            
           
            model = slim.conv2d(model, 1024, [16, 16], padding='VALID', scope='conv2', stride=16) # 32*H*W -> 1024*H/16*W/16
            
            model = slim.conv2d_transpose(model, self.input_shape[2], [16, 16], stride=16, padding='VALID', activation_fn=None, scope='deconv_1')
            # assert 1 == 0, str(model)
            return model

    @model_property
    def loss(self):
        y = tf.reshape(self.y, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        
        # For a fancy tensorboard summary: put the input, label and model side by side (sbs) for a fancy image summary:
        # sbs = tf.concat(2, [x, y, model])
        # tf.summary.image(sbs.op.name, sbs, max_outputs=3, collections=["training summary"])
        return digits.mse_loss(self.inference, y)
