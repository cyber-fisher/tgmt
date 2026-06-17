#!/usr/bin/env python3

from common import parse_args, run_surface_pipeline


def main() -> None:
    options = parse_args(
        description="Surface VTP tree ISM simplifier that intentionally keeps only one isocontour component.",
        mode_name="surface_vtp_tree_ism_single_component",
        single_component_only=True,
    )
    run_surface_pipeline(options)


if __name__ == "__main__":
    main()
