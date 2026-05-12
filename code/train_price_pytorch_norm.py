from train_price_pytorch_extended import NORM_MODEL_TYPES, main


if __name__ == "__main__":
    main(
        allowed_model_types=NORM_MODEL_TYPES,
        default_model_type="lstm_pre_norm",
        description="Train LSTM/BiLSTM pre/post LayerNorm stock-price baselines.",
    )
