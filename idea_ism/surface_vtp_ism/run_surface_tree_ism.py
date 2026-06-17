#!/usr/bin/env python3

from common import parse_args, run_surface_pipeline


def main() -> None:
    options = parse_args(
        description="Surface VTP tree ISM simplifier that keeps all isocontour components.",
        mode_name="surface_vtp_tree_ism",
        single_component_only=False,
    )
    run_surface_pipeline(options)


if __name__ == "__main__":
    main()
