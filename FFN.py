import tensorflow as tf 
import numpy as np

class FFN(tf.keras.models.Model):
    def __init__(self, embedding_dim: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        self.dense_4_layer = tf.keras.layers.Dense(embedding_dim*4, use_bias=False, name='dense_4_layer')
        self.dense_1_layer = tf.keras.layers.Dense(embedding_dim, use_bias=False, name='dense_1_layer')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, attention_output: tf.Tensor, training: bool) -> tf.Tensor:
        tensor = self.dense_4_layer(attention_output)
        tensor = tf.nn.relu(tensor)
        tensor = self.dropout_layer(tensor, training=training)
        tensor = self.dense_1_layer(tensor)
        return tensor   

if __name__ == '__main__':
    # テスト用にランダムなテンソルを生成
    batch_size = 2
    length = 10
    embedding_dim = 64
    attention_output = tf.random.uniform((batch_size, length, embedding_dim), dtype=tf.float32)

    # FFNクラスのインスタンスを作成
    ffn_layer = FFN(embedding_dim=embedding_dim, dropout_rate=0.1)

    # トレーニングモードでの出力を計算
    output = ffn_layer(attention_output, training=True)

    # 結果を表示
    print("Input shape:", attention_output.shape)
    print("Output shape:", output.shape)
    print("Output tensor:", output)