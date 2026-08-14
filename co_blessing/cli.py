from __future__ import annotations

import argparse

from .analysis import analyze_features, analyze_induce, analyze_masks, analyze_vact_curves
from .config import apply_overrides, load_config
from .evaluation import evaluate
from .reporting import compare_results
from .reproduce import reproduce
from .training import train


def _common_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--output-root", help="Override output.root")
    parser.add_argument("--device", help="Override device, e.g. cuda:0 or cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m co_blessing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one configured model")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume")
    _common_overrides(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Run the paper evaluation protocol")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    _common_overrides(eval_parser)

    analyze_parser = subparsers.add_parser("analyze", help="Reproduce mechanism analyses")
    analyze_parser.add_argument("--task", required=True, choices=["features", "mask", "induce"])
    analyze_parser.add_argument("--config")
    analyze_parser.add_argument("--checkpoint")
    analyze_parser.add_argument("--runs", nargs="+")
    analyze_parser.add_argument("--output", required=True)
    _common_overrides(analyze_parser)

    compare_parser = subparsers.add_parser("compare", help="Compare evaluation JSON files to Table 2")
    compare_parser.add_argument("--results", nargs="+", required=True)
    compare_parser.add_argument("--output", required=True)

    reproduce_parser = subparsers.add_parser("reproduce", help="Run a sequential reproduction manifest")
    reproduce_parser.add_argument("--manifest", required=True)
    _common_overrides(reproduce_parser)
    return parser


def _configured(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    return apply_overrides(
        config,
        data_root=getattr(args, "data_root", None),
        output_root=getattr(args, "output_root", None),
        device=getattr(args, "device", None),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        print(train(_configured(args), resume=args.resume))
    elif args.command == "evaluate":
        print(evaluate(_configured(args), args.checkpoint))
    elif args.command == "analyze":
        if args.task == "induce":
            if not args.runs:
                raise SystemExit("--runs is required for induce analysis")
            print(analyze_induce(args.runs, args.output))
        elif args.task == "features" and args.runs:
            print(analyze_vact_curves(args.runs, args.output))
        else:
            if not args.config or not args.checkpoint:
                raise SystemExit("--config and --checkpoint are required for feature/mask analysis")
            config = _configured(args)
            if args.task == "features":
                print(analyze_features(config, args.checkpoint, args.output))
            else:
                print(analyze_masks(config, args.checkpoint, args.output))
    elif args.command == "compare":
        print(compare_results(args.results, args.output))
    elif args.command == "reproduce":
        print(
            reproduce(
                args.manifest,
                data_root=args.data_root,
                output_root=args.output_root,
                device=args.device,
            )
        )
    else:
        raise AssertionError(args.command)
    return 0
