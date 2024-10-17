### Mercari Price Suggestion Challenge

This project is based on Kaggle's **Mercari Price Suggestion Challenge**, where the goal is to predict the prices of items listed on the Mercari platform. Mercari allows users to set prices freely, which can lead to pricing that is either too high (resulting in unsold items) or too low (causing sellers to lose money). The challenge focuses on using machine learning models to provide better price suggestions, evaluated using the Root Mean Squared Logarithmic Error (RMSLE) to improve accuracy.

### Approach

The model uses the `item_description` column as input to GPT’s Encoder. The input text is processed through layers such as **Attention** and **MLP** (Multi-Layer Perceptron). We assume that the meaning of the entire sentence is encapsulated in the final token of the sequence. This final token is then passed through a simple neural network to predict the price.

### Model Performance

The current Mean Squared Error (MSE) is around **600**, indicating there’s still room for improvement.

### How to Use

1. Modify `training.py` by setting `inputs` as a list of item descriptions and `targets` as a list of corresponding prices.
2. The model will then be ready to predict prices.

