import tensorflow as tf
import numpy as np

class ResidualNormalizationWrapper(tf.keras.models.Model):
    def __init__(self, layer: tf.keras.layers.Layer, dropout_rate: float,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.layer = layer
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs: tf.Tensor, training: bool, *args, **kwargs):
        x = self.layer_norm(inputs)
        x = self.layer(x, training=training, *args, **kwargs)
        x = self.dropout_layer(x, training=training)
        x += inputs
        return x
