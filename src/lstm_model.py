import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


class LSTMModel:
    """
    Modelo LSTM para prediccion de series temporales de ETFs
    """

    def __init__(self, window_size: int = 60, lstm_units: int = 256, dropout: float = 0.3, learning_rate: float = 1e-4):

        """
        Args: 
            window_size: Tamano de la ventana temporal
            
        """
        
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self._build()
        
    def _build(self)-> Sequential:
        model = Sequential([
            Conv1D(filters=128,kernel_size=5, stride=1, 
                activation="relu", padding="causal", 
                input_shape=[self.window_size, 1]),
            LSTM(self.lstm_units, return_sequencies=True),
            Dropout(self.dropout),
            LSTM(self.lstm_units, return_sequencies= False),
            Dropout(self.dropout),
            Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(1)
        ])

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]   
        )

        return model

    