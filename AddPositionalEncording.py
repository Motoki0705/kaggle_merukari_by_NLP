import tensorflow as tf
import numpy as np

class AddPositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        batch_size, length, depth = tf.cast(tf.unstack(tf.shape(inputs)), fl_type)
        
        depth_counter = tf.range(depth, dtype=fl_type) #[0, 1, ..., depth-1]
        depth_matrix = tf.pow(10000.0, depth_counter / depth) #[100000**(0/depth), 100000**(1/depth), ..., 100000**((depth-1)/depth)]

        length_counter = tf.range(length, dtype=fl_type) #[0, 1, ..., length-1]
        length_matrix = tf.tile(tf.expand_dims(length_counter, 1), [1, depth]) #[length, depth]
        
        sin = tf.sin(length_matrix/depth_matrix) #[length, depth]
        cos = tf.cos(length_matrix/depth_matrix) #[length, depth]
        
        pos_matrix = tf.where(depth_counter % 2 == 0, sin, cos) #[length, depth]
    
        return inputs + pos_matrix
    
# テストコード
if __name__ == "__main__":
    # テスト用のダミー入力データ（batch_size=2, length=10, depth=16）
    test_input = tf.random.uniform((2, 10, 16), dtype=tf.float32)

    # AddPositionalEncoding レイヤーのインスタンスを作成
    positional_encoding_layer = AddPositionalEncoding()

    # 位置エンコーディングを追加
    output = positional_encoding_layer(test_input)

    # 結果を表示
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
    print("Output with positional encoding added:", output)