"""Train multiple DINOv2 source domains and evaluate one unseen target domain."""

import argparse

from dinov2_engine import add_common_arguments, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--target-domain", "--target", dest="target_domain", required=True)
    parser.add_argument(
        "--source-domains", nargs="+", required=True,
        help="Explicit source-domain list, e.g. HYRD PAPILA DRISHTI_GS.",
    )
    args = parser.parse_args()
    run_experiment("msd", args.source_domains, args.target_domain, args)


if __name__ == "__main__":
    main()
