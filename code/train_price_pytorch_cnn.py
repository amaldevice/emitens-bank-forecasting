from train_price_pytorch_extended import CNN_MODEL_TYPES, main


if __name__ == "__main__":
    main(
        allowed_model_types=CNN_MODEL_TYPES,
        default_model_type="cnn_lstm",
        description="Train CNN-LSTM/CNN-BiLSTM stock-price baselines.",
    )
