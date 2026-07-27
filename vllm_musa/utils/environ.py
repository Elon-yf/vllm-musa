# Copied and adapted from: https://github.com/sgl-project/sglang
import os
import warnings
from contextlib import contextmanager
from typing import Any


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y", "on"]:
            return True
        if value in ["false", "0", "no", "n", "off"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        return int(value)


class Envs:
    VLLM_MUSA_CUSTOM_OP_USE_NATIVE = EnvBool(False)
    VLLM_MUSA_FUSED_ADD_RMSNORM = EnvBool(True)
    # Enable the MUSA CAR-RMSNorm graph rewrite and fused runtime path.
    VLLM_MUSA_FUSED_AR_RMSNORM = EnvBool(False)
    # Opt in to exchanging Graph-input IPC handles. Eager execution and the
    # default Graph path keep using the fixed staging buffer.
    VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT = EnvBool(False)
    # Direct graph-input IPC was faster through 512 KiB and regressed at
    # 640/960 KiB in paired TP2 tests; larger inputs keep the staging path.
    VLLM_MUSA_FUSED_AR_RMSNORM_GRAPH_REGISTERED_INPUT_MAX_BYTES = EnvInt(
        512 * 1024
    )
    # Opt-in detailed graph-candidate diagnostics for CAR-RMSNorm fusion.
    # Keep disabled by default so normal inference has no per-node log volume.
    VLLM_MUSA_FUSED_AR_RMSNORM_DEBUG = EnvBool(False)
    VLLM_MUSA_ENABLE_JIT_TOPK = EnvBool(True)
    VLLM_MUSA_SEEDED_MULTINOMIAL = EnvBool(True)
    VLLM_MUSA_RESHAPE_CACHE_FLASH = EnvBool(True)


envs = Envs()
