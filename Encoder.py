import tensorflow as tf
import numpy as np
from WordEmbedding import WordEmbedding
from ResidualNormalizationWrapper import ResidualNormalizationWrapper
from AddPositionalEncording import AddPositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from FFN import FFN

class Encoder(tf.keras.models.Model):
    def __init__(self, d_model: int, embedding_dim: int, head_num: int, batch_size: int, dropout_rate: float, N: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.d_model = d_model
        self.batch_size = batch_size
        
        self.WordEmbedding = WordEmbedding(d_model, embedding_dim)
        self.AddPositional = AddPositionalEncoding()
        self.MultiHeadAttention = MultiHeadAttention(embedding_dim, head_num)
        self.FFN = FFN(embedding_dim, dropout_rate)
        self.M_ResidualNormalizationWapper = ResidualNormalizationWrapper(self.MultiHeadAttention, dropout_rate)
        self.F_ResidualNormalizationWapper = ResidualNormalizationWrapper(self.FFN, dropout_rate)
        
        self.output_layer_1 = tf.keras.layers.Dense(int(embedding_dim/2), activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.output_layer_2 = tf.keras.layers.Dense(1, activation='linear')
        
    def call(self, inputs: list, training: bool):
        query, mask = self.WordEmbedding.call(inputs) #[batch_size, d_model, embedding_dim]
        query = self.AddPositional(query) #[batch_size, d_model, embedding_dim]
        attention_mask = self.create_mask(mask)
    
        for _ in range(self.N):
            x = self.M_ResidualNormalizationWapper(inputs=query, attention_mask=attention_mask, training=training) #[batch_size, d_model, embedding_dim]
            x = self.F_ResidualNormalizationWapper(x, training=training) #[batch_size, d_model, embedding_dim]
            
        x = x[:, -1, :] #[batch_size, embedding_dim]
        x = self.output_layer_1(x)
        x = self.batch_norm(x, training=training)
        output = self.output_layer_2(x)
        
        return output
    
    def create_mask(self, mask: tf.Tensor):
        attention_mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, self.d_model])
        return attention_mask
 