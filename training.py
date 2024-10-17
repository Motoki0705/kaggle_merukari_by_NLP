import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Encoder import Encoder

df = pd.read_csv('train.tsv', sep='\t')

df = df.dropna(subset=['item_description'])

inputs = df['item_description']
targets = df['price']

  
if 1:   
    
    # モデルのインスタンス作成
    d_model = 128
    embedding_dim = 64
    head_num = 8
    batch_size = 128
    dropout_rate = 0.1
    n = 6    
    x_train, x_test, t_train, t_test = train_test_split(inputs, targets, test_size=0.1, random_state=0)

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
    for batch_idx, (batch_inputs, batch_targets) in tqdm((enumerate(create_random_batches(x_train, t_train, batch_size)))):
        train_step(np.array(batch_inputs), np.array(batch_targets))
        
        if (batch_idx % 20) == 0:
            print(f'Batch {batch_idx+1}, train_loss: {train_loss.result():.3f}')
    
    # テストデータのバッチ処理（こちらはシャッフル不要だが同様にバッチ処理）
    for batch_idx, (batch_inputs, batch_targets) in tqdm(enumerate(create_random_batches(x_test, t_test, batch_size))):
        test_step(np.array(batch_inputs), np.array(batch_targets))
        
        if (batch_idx % 20) == 0:
            print(f'Batch {batch_idx+1}, test_loss: {test_loss.result():.3f}')

    print(f'Epoch {epoch+1}, train_loss: {train_loss.result():.3f}, test_loss: {test_loss.result():.3f}')
