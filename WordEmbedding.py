import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
import pickle

class WordEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        
        # BERTのトークナイザを初期化
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 埋め込み層を作成
        vocab_size = self.tokenizer.vocab_size
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    def call(self, inputs: list):
        # リスト形式でなければ、リストに変換
        if isinstance(inputs, np.ndarray):
            inputs = inputs.tolist()  # numpy配列ならリストに変換
        elif isinstance(inputs, str):
            inputs = [inputs]  # 文字列なら単一要素のリストに変換
        elif not isinstance(inputs, list):
            raise ValueError("Input should be a list of strings or a numpy array.")
        
        # トークナイザでトークン化
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            inputs,  # ここではリスト形式の入力が渡されます
            max_length=self.d_model,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # 埋め込みベクトルを取得
        vector_matrix = self.embedding_layer(tokenized_inputs['input_ids'])
        
        # 埋め込みベクトルとアテンションマスクを返す
        return vector_matrix, tokenized_inputs['attention_mask']
    
    # 埋め込み行列を取得するメソッド
    def get_embedding_matrix(self):
        return self.embedding_layer.weights[0].numpy()

if __name__ == '__main__':
        
    with open(r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\item_description\x.pkl', 'rb') as f:
        inputs = pickle.load(f).tolist()
    with open (r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\item_description\target.pkl', 'rb') as f:
        targets = pickle.load(f).tolist()
        
    word = WordEmbedding(50, 100)   
    
    print(inputs[:128])
    inputs = tf.data.Dataset.from_tensor_slices(inputs).batch(128)
    for input in inputs:
        input = input.numpy().tolist()
        print(input)
        break