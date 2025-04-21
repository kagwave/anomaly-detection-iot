import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assign callback manually to avoid Pylance squiggles
EarlyStopping = tf.keras.callbacks.EarlyStopping

def create_lstm_attention_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x_last = tf.keras.layers.LSTM(64)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x_last)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_lstm_model(features, labels):
    df = features.copy()
    y = labels.dropna().astype(int)
    X = df.loc[y.index]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape to 3D: (samples, timesteps, features_per_step)
    # Here, each sample is treated as a single-timestep sequence
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, shuffle=False)

    model = create_lstm_attention_model(input_shape=(1, X_scaled.shape[1]))

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    preds = model.predict(X_seq).flatten()
    return preds, model