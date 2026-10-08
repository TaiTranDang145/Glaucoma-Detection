"""Train one DINOv2 source domain and evaluate one unseen target domain."""

from dinov2_engine import build_arg_parser, run_experiment


def main():
    parser = build_arg_parser(__doc__)
    parser.add_argument("--source", required=True, help="Single source domain, e.g. HYRD.")
    args = parser.parse_args()
    run_experiment("ssd", [args.source], args.target_domain, args)


if __name__ == "__main__":
    main()
