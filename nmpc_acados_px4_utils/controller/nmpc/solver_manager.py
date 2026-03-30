"""Utilities for platform-aware acados solver generation and caching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from quad_platforms import PLATFORM_REGISTRY, PlatformType

from .acados_model import QuadrotorEulerModel
from .generate_nmpc import QuadrotorEulerErrMPC


MODEL_NAME = "holybro_euler_err"
DEFAULT_HORIZON = 2.0
DEFAULT_NUM_STEPS = 50
STAMP_VERSION = 1


@dataclass(frozen=True)
class SolverSpec:
    """Expected solver configuration for the generated acados artifacts."""

    stamp_version: int
    model_name: str
    platform: str
    mass: float
    horizon: float
    num_steps: int
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stamp_version": self.stamp_version,
            "model_name": self.model_name,
            "platform": self.platform,
            "mass": self.mass,
            "horizon": self.horizon,
            "num_steps": self.num_steps,
            "source_hash": self.source_hash,
        }


def generated_root() -> Path:
    return Path(__file__).resolve().parent / "acados_generated_files"


def codegen_dir() -> Path:
    return generated_root() / f"{MODEL_NAME}_mpc_c_generated_code"


def json_path() -> Path:
    return generated_root() / f"{MODEL_NAME}_mpc_acados_ocp.json"


def stamp_path() -> Path:
    return generated_root() / f"{MODEL_NAME}_solver_stamp.json"


def resolve_platform_mass(platform: PlatformType | str | None) -> float:
    if platform is None:
        raise ValueError("platform must be provided when mass is not specified")
    platform_type = platform if isinstance(platform, PlatformType) else PlatformType(platform)
    return PLATFORM_REGISTRY[platform_type]().mass


def resolve_solver_spec(
    *,
    platform: PlatformType | str | None = None,
    mass: float | None = None,
    horizon: float = DEFAULT_HORIZON,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> SolverSpec:
    platform_value = (
        platform.value if isinstance(platform, PlatformType) else platform
    ) or "custom"
    resolved_mass = mass if mass is not None else resolve_platform_mass(platform)
    return SolverSpec(
        stamp_version=STAMP_VERSION,
        model_name=MODEL_NAME,
        platform=str(platform_value),
        mass=float(resolved_mass),
        horizon=float(horizon),
        num_steps=int(num_steps),
        source_hash=_source_hash(),
    )


def load_solver_stamp() -> dict[str, Any] | None:
    path = stamp_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_solver_stamp(spec: SolverSpec) -> None:
    generated_root().mkdir(parents=True, exist_ok=True)
    stamp_path().write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n")


def missing_solver_artifacts() -> list[str]:
    missing: list[str] = []
    required_paths = {
        "codegen_dir": codegen_dir(),
        "json": json_path(),
        "header": codegen_dir() / f"acados_solver_{MODEL_NAME}.h",
        "shared_lib": codegen_dir() / f"libacados_ocp_solver_{MODEL_NAME}.so",
    }
    for label, path in required_paths.items():
        if not path.exists():
            missing.append(f"{label}:{path}")

    if not any(codegen_dir().glob("acados_ocp_solver_pyx*.so")):
        missing.append(f"python_solver:{codegen_dir() / 'acados_ocp_solver_pyx*.so'}")
    return missing


def describe_solver_state(spec: SolverSpec) -> str:
    parts: list[str] = []
    current_stamp = load_solver_stamp()
    if current_stamp is None:
        parts.append("stamp file missing or unreadable")
    else:
        for field in ("stamp_version", "model_name", "platform", "mass", "horizon", "num_steps", "source_hash"):
            if not _stamp_field_matches(current_stamp, spec, field):
                parts.append(
                    f"{field} changed (want={spec.to_dict()[field]!r}, have={current_stamp.get(field)!r})"
                )

    missing = missing_solver_artifacts()
    if missing:
        parts.append("missing artifacts: " + ", ".join(missing))

    return "; ".join(parts) if parts else "up to date"


def ensure_solver_artifacts(
    *,
    platform: PlatformType | str | None = None,
    mass: float | None = None,
    horizon: float = DEFAULT_HORIZON,
    num_steps: int = DEFAULT_NUM_STEPS,
    force: bool = False,
) -> dict[str, Any]:
    spec = resolve_solver_spec(platform=platform, mass=mass, horizon=horizon, num_steps=num_steps)
    reason = "force requested" if force else describe_solver_state(spec)

    if not force and reason == "up to date":
        print(
            "[acados] Solver cache is current for "
            f"platform={spec.platform}, mass={spec.mass:.6f}, horizon={spec.horizon}, num_steps={spec.num_steps}."
        )
        return {"generated": False, "reason": reason, "spec": spec}

    print(
        "[acados] Regenerating solver for "
        f"platform={spec.platform}, mass={spec.mass:.6f}, horizon={spec.horizon}, num_steps={spec.num_steps}."
    )
    print(f"[acados] Reason: {reason}")

    model = QuadrotorEulerModel(mass=spec.mass)
    QuadrotorEulerErrMPC(
        generate_c_code=True,
        quadrotor=model,
        horizon=spec.horizon,
        num_steps=spec.num_steps,
    )
    write_solver_stamp(spec)
    print(f"[acados] Updated solver stamp at {stamp_path()}")
    return {"generated": True, "reason": reason, "spec": spec}


def _source_hash() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parent.parents[2]
    for path in _tracked_source_files():
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tracked_source_files() -> list[Path]:
    module_dir = Path(__file__).resolve().parent
    package_root = module_dir.parents[2]
    return [
        module_dir / "acados_model.py",
        module_dir / "generate_nmpc.py",
        module_dir / "solver_manager.py",
        package_root / "generate_solver.py",
        package_root / "ensure_solver.py",
    ]


def _stamp_field_matches(stamp: dict[str, Any], spec: SolverSpec, field: str) -> bool:
    expected = spec.to_dict()[field]
    actual = stamp.get(field)
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected
