import tensorflow as tf
import numpy as np

class MultiHeadAttention(tf.keras.models.Model):
    
    def __init__(self, embedding_dim: int, head_num: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        
        self.q_dense_layer = tf.keras.layers.Dense(embedding_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(embedding_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(embedding_dim, use_bias=False, name='v_dense_layer')
        
        self.output_dense_layer = tf.keras.layers.Dense(embedding_dim, use_bias=False, name='output_dense_layer')
    
    def call(self, query: tf.Tensor, attention_mask: tf.Tensor, training: bool) -> tf.Tensor:
        
        q = self.q_dense_layer(query)
        k = self.k_dense_layer(query)
        v = self.v_dense_layer(query)
        
        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)
        
        # Attention maskの形状を変更して、各ヘッドに適用
        attention_mask = tf.expand_dims(attention_mask, axis=1)  # Shape: [batch_size, 1, d_model, d_model]
        
        q = tf.divide(q, tf.sqrt(tf.cast(self.embedding_dim, tf.float32)))
        logit = tf.matmul(q, k, transpose_b=True)
        logit += tf.cast(attention_mask, tf.float32) * -1e9  # マスクで大きな負の値を適用
        
        attention_weight = tf.nn.softmax(logit, axis=-1)
        context_vector = tf.matmul(attention_weight, v)
        
        attention_output = self._concat_head(context_vector)
        attention_output = self.output_dense_layer(attention_output)
        
        return attention_output
        
    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('split_head'):
            batch_size, length, embedding_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, embedding_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])
        
    def _concat_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('concat_head'):
            batch_size, head_num, length, head_embedding_dim = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, head_num * head_embedding_dim])


if __name__ == "__main__":
    # テスト用のダミーデータを作成
    batch_size = 2
    length = 10
    embedding_dim = 64
    head_num = 8
    dropout_rate = 0.1
    
    # ランダムなテンソルデータ
    query = tf.random.uniform((batch_size, length, embedding_dim))
    memory = tf.random.uniform((batch_size, length, embedding_dim))
    attention_mask = tf.ones((batch_size, length, length))  # マスクの形を [batch_size, length, length] に変更
    
    # MultiHeadAttentionレイヤーのインスタンス作成
    mha = MultiHeadAttention(embedding_dim=embedding_dim, head_num=head_num, dropout_rate=dropout_rate)
    
    # 出力を計算
    output = mha(query, attention_mask, training=True)
    
    # 結果を表示
    print("Output shape:", output.shape)
    print("Output:", output)
