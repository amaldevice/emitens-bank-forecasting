from train_sota_common import parse_args, run_pipeline


def main() -> None:
    config = parse_args(
        default_data_dir="data",
        default_output_root="results_sota_data",
        use_macro=False,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
