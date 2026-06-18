"""CLI entrypoint for training runs."""

from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="TTA3DCache training entrypoint")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment config file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
