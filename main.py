import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
    
with open(r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\item_description\x.pkl', 'rb') as f:
    inputs = pickle.load(f).tolist()
with open (r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\item_description\target.pkl', 'rb') as f:
    targets = pickle.load(f).tolist()
    
if 1:   
    # モデルのインスタンス作成
    d_model = 128
    embedding_dim = 64
    head_num = 8
    batch_size = 128
    dropout_rate = 0.1
    n = 6    
    x_train, x_test, t_train, t_test = train_test_split(inputs, targets, test_size=0.1, random_state=0)

    train_datasets = tf.data.Dataset.from_tensor_slices((x_train, t_train)).shuffle(len(x_train)).batch(batch_size)
    test_datasets = tf.data.Dataset.from_tensor_slices((x_test, t_test)).batch(batch_size)

    encoder = Encoder(d_model, embedding_dim, head_num, batch_size, dropout_rate, n)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            
            pred = encoder.call(inputs, training=True)
            loss = loss_object(targets, pred)
        
        gradients = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
        
        train_loss(loss)
        
    def test_step(inputs, targets):
        predictions = encoder(inputs, training=False)
        t_loss = loss_object(targets, predictions)
        test_loss(t_loss)

# データをシャッフルしてバッチごとに手動で処理
def create_random_batches(data, targets, batch_size):
    # データをシャッフル
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    # シャッフルされたデータとターゲットを取得
    shuffled_data = [data[i] for i in indices]
    shuffled_targets = [targets[i] for i in indices]
    
    # バッチを生成
    for i in range(0, len(shuffled_data), batch_size):
        yield shuffled_data[i:i + batch_size], shuffled_targets[i:i + batch_size]

epochs = 100
for epoch in range(epochs):
    train_loss.reset_state()
    test_loss.reset_state()

    # 訓練データのバッチ処理（ランダムにシャッフルしてバッチ化）
    for batch_idx, (batch_inputs, batch_targets) in enumerate(create_random_batches(x_train, t_train, batch_size)):
        train_step(np.array(batch_inputs), np.array(batch_targets))
        
        if (batch_idx % 20) == 0:
            print(f'Batch {batch_idx+1}, train_loss: {train_loss.result():.3f}')
    
    # テストデータのバッチ処理（こちらはシャッフル不要だが同様にバッチ処理）
    for batch_idx, (batch_inputs, batch_targets) in enumerate(create_random_batches(x_test, t_test, batch_size)):
        test_step(np.array(batch_inputs), np.array(batch_targets))
        
        if (batch_idx % 20) == 0:
            print(f'Batch {batch_idx+1}, test_loss: {test_loss.result():.3f}')

    print(f'Epoch {epoch+1}, train_loss: {train_loss.result():.3f}, test_loss: {test_loss.result():.3f}')


        
        

