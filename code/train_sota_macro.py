from train_sota_common import parse_args, run_pipeline


def main() -> None:
    config = parse_args(
        default_data_dir="data_with_macro",
        default_output_root="results_sota_macro",
        use_macro=True,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
