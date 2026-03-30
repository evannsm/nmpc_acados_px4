#!/usr/bin/env python3
"""Standalone script to generate the acados NMPC solver C code.

This must be run before building nmpc_acados_px4_cpp.
The generated files are written to:
  nmpc_acados_px4_utils/controller/nmpc/acados_generated_files/

Usage:
  python3 generate_solver.py [--platform {sim,hw}] [--mass MASS] [--horizon H] [--num-steps N]

Defaults use the GZ X500 simulation platform mass.
"""

import argparse

from quad_platforms import PlatformType

from nmpc_acados_px4_utils.controller.nmpc.solver_manager import (
    DEFAULT_HORIZON,
    DEFAULT_NUM_STEPS,
    ensure_solver_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description="Generate acados NMPC solver C code")
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
        help="Vehicle mass in kg. Overrides the mass implied by --platform.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=DEFAULT_HORIZON,
        help=f"MPC horizon in seconds (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of horizon steps (default: {DEFAULT_NUM_STEPS})",
    )
    args = parser.parse_args()

    result = ensure_solver_artifacts(
        platform=args.platform,
        mass=args.mass,
        horizon=args.horizon,
        num_steps=args.num_steps,
        force=True,
    )
    spec = result["spec"]
    print(
        f"\n[generate_solver] platform={spec.platform}, mass={spec.mass} kg, "
        f"horizon={spec.horizon} s, num_steps={spec.num_steps}"
    )
    print("[generate_solver] Done. You can now build nmpc_acados_px4_cpp.\n")


if __name__ == "__main__":
    main()
