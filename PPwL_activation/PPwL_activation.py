"""
Created on Jun 17 2018

@author: lpupp

# Sources:
Constraints:
    https://keras.io/constraints/
    https://github.com/keras-team/keras/blob/master/keras/constraints.py
Layers:
    https://keras.io/layers/writing-your-own-keras-layers/
Keras:
    https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L18
"""
from keras import initializers
#from keras import regularizers
from keras import constraints

import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import Constraint

class PPwL(Layer):
    """Parametric Piecewise linear.
    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = (x - 1) * beta +1 for x > 1`,
    `f(x) = x else`,
    where `alpha` and `beta` are learned arrays with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the alpha parameter.
        beta_initializer:  initializer function for the beta parameter.
        alpha_regularizer: regularizer for the weights.
        alpha_constraint: constraint for the weights.

    # References
        - TBD
    """
    def __init__(self, alpha_initializer='ones',
                 beta_initializer='ones',
                 #alpha_regularizer=None,
                 #beta_regularizer=None,
                 alpha_constraint=None,
                 beta_constraint=None,
                 alpha_trainable = True,
                 beta_trainable = True,
                 **kwargs):
        super(PPwL, self).__init__(**kwargs)

        self.alpha_initializer = initializers.get(alpha_initializer)
        #self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.alpha_trainable = alpha_trainable

        self.beta_initializer = initializers.get(beta_initializer)
        #self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.beta_trainable = beta_trainable

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        # Create a trainable weight variable for this layer.
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     #regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint,
                                     trainable=self.alpha_trainable
                                     )

        self.beta = self.add_weight(shape=param_shape,
                                     name='beta',
                                     initializer=self.beta_initializer,
                                     #regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint,
                                     trainable=self.beta_trainable
                                     )

        super(PPwL, self).build(input_shape)

    def call(self, x):
        pos = tf.where(x > 1, (x - 1) * self.beta + 1, x)
        return tf.where(x < 0, self.alpha * x, pos)

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            #'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'alpha_trainable': self.alpha_trainable,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            #'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'beta_trainable': self.beta_trainable
        }
        base_config = super(PPwL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class min_max_bound(Constraint):
    """min_max_bound weight constraint.
    Binds the weights between `min_value` and `max_value` (by clipping).
    # Arguments
        min_value: the minimum for the incoming weights.
        max_value: the maximum for the incoming weights.
        esp: clipping tolerance
        _axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """
    def __init__(self, min_value=0.0,
                 max_value=1.0,
                 eps=0.,
                 axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps
        self.axis = axis

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'esp': self.eps,
                'axis': self.axis}
