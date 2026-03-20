#!/usr/bin/env python3
"""Standalone script to generate the acados NMPC solver C code.

This must be run before building nmpc_acados_px4_cpp.
The generated files are written to:
  nmpc_acados_px4_utils/controller/nmpc/acados_generated_files/

Usage:
  python3 generate_solver.py [--mass MASS] [--horizon H] [--num-steps N]

Defaults use the GZ X500 simulation platform mass (1.95 kg).
"""

import argparse
from nmpc_acados_px4_utils.controller.nmpc import QuadrotorEulerModel, QuadrotorEulerErrMPC


def main():
    parser = argparse.ArgumentParser(description="Generate acados NMPC solver C code")
    parser.add_argument("--mass", type=float, default=1.95,
                        help="Vehicle mass in kg (default: 1.95, GZ X500 sim)")
    parser.add_argument("--horizon", type=float, default=2.0,
                        help="MPC horizon in seconds (default: 2.0)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of horizon steps (default: 50)")
    args = parser.parse_args()

    print(f"\n[generate_solver] mass={args.mass} kg, horizon={args.horizon} s, "
          f"num_steps={args.num_steps}")

    model = QuadrotorEulerModel(mass=args.mass)
    QuadrotorEulerErrMPC(
        generate_c_code=True,
        quadrotor=model,
        horizon=args.horizon,
        num_steps=args.num_steps,
    )
    print("[generate_solver] Done. You can now build nmpc_acados_px4_cpp.\n")


if __name__ == "__main__":
    main()
