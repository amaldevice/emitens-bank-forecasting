from train_price_pytorch_extended import ATTENTION_MODEL_TYPES, main


if __name__ == "__main__":
    main(
        allowed_model_types=ATTENTION_MODEL_TYPES,
        default_model_type="lstm_attention",
        description="Train LSTM/BiLSTM temporal-attention stock-price baselines.",
    )
