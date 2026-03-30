#!/usr/bin/env python3
"""Ensure the acados NMPC solver matches the requested platform and formulation."""

from __future__ import annotations

import argparse

from quad_platforms import PlatformType

from nmpc_acados_px4_utils.controller.nmpc.solver_manager import (
    DEFAULT_HORIZON,
    DEFAULT_NUM_STEPS,
    ensure_solver_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure the generated acados solver matches the requested platform/mass."
    )
    parser.add_argument(
        "--platform",
        type=PlatformType,
        choices=list(PlatformType),
        default=PlatformType.SIM,
        help="Platform whose mass should be encoded in the solver (default: sim).",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=None,
        help="Explicit vehicle mass in kg. Overrides --platform mass lookup.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=DEFAULT_HORIZON,
        help=f"MPC horizon in seconds (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of horizon steps (default: {DEFAULT_NUM_STEPS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if the solver stamp already matches.",
    )
    args = parser.parse_args()

    ensure_solver_artifacts(
        platform=args.platform,
        mass=args.mass,
        horizon=args.horizon,
        num_steps=args.num_steps,
        force=args.force,
    )


if __name__ == "__main__":
    main()
