# LSTM Text Generation Model Using BOW Method

A deep learning natural language processing (NLP) project focused on sequence-to-sequence modeling and next-word text prediction. Built with TensorFlow and Keras, this model utilizes a stacked Long Short-Term Memory (LSTM) recurrent neural network architecture trained on localized product review data to dynamically auto-complete text strings.

## Model Architecture

The deep learning pipeline is constructed using a sequential architecture with optimized data flow:

*   **Embedding Layer:** Maps the high-dimensional sparse vocabulary vectors into a dense, continuous 50-dimensional vector space.
*   **Stacked LSTM Layers:** Two consecutive 100-unit Long Short-Term Memory layers. The first layer passes its full sequence vectors (`return_sequences=True`) to the second layer to capture complex, long-range contextual semantic dependencies.
*   **Dense / Dropout Regularization:** A fully connected 50-unit ReLU layer paired with a 30% Dropout rate to enforce structural robustness and prevent over-fitting during training.
*   **Softmax Output Layer:** Maps network weights against the complete vocabulary distribution matrix to calculate precise next-word probabilities.

## Data Engineering & Pipelines

*   **Sliding-Window Tokenization:** Reviews are preprocessed using a custom sliding-window sequence matrix. The input reads sequences of text tokens to dynamically predict the subsequent trailing word token.
*   **Sequence Padding:** Sequences are dynamically normalized and left-padded via `pad_sequences` to feed uniform input shapes into the network array.
*   **Target Vectorization:** Training targets are mapped using categorical cross-entropy loss with one-hot encoded label distributions.

## Tech Stack

*   **Frameworks:** TensorFlow, Keras
*   **Data Science Utilities:** Pandas, NumPy
*   **Environment:** Jupyter Notebook / Google Colab (GPU accelerated)
