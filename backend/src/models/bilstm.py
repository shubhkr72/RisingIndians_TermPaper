from tensorflow.keras import Input, Model, layers


def build_bilstm_cnn(vocab_size: int, max_len: int = 300, embedding_dim: int = 100) -> Model:
    inp = Input(shape=(max_len,), name="token_input")
    emb = layers.Embedding(vocab_size, embedding_dim, name="embedding")(inp)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_1"
    )(emb)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64), name="bilstm_2")(x)
    x = layers.Dropout(0.3)(x)

    c = layers.Conv1D(128, 3, activation="relu", name="conv1d")(emb)
    c = layers.GlobalMaxPooling1D(name="global_max_pool")(c)
    c = layers.Dense(64, activation="relu", name="cnn_dense")(c)

    fused = layers.Concatenate(name="fusion")([x, c])
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    out = layers.Dense(1, activation="sigmoid", name="output")(fused)

    return Model(inp, out, name="bilstm_cnn")
