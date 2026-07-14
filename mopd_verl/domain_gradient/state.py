"""Capture and restore state touched by audit forward/backward replay."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl.utils.device import get_torch_device


@dataclass(frozen=True)
class RngState:
    python: object
    numpy: tuple[Any, ...]
    cpu: torch.Tensor
    accelerator: tuple[torch.Tensor, ...]

    @classmethod
    def capture(cls) -> "RngState":
        device = get_torch_device()
        getter = getattr(device, "get_rng_state_all", None)
        accelerator = tuple(state.cpu().clone() for state in getter()) if callable(getter) else tuple()
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(),
            cpu=torch.get_rng_state().clone(),
            accelerator=accelerator,
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if not self.accelerator:
            return
        setter = getattr(get_torch_device(), "set_rng_state_all", None)
        if not callable(setter):
            raise RuntimeError("Accelerator RNG state was captured but cannot be restored.")
        setter(list(self.accelerator))


@dataclass(frozen=True)
class AuditState:
    rng: RngState
    parameters: tuple[torch.nn.Parameter, ...]
    gradients: tuple[torch.Tensor | None, ...]
    buffers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    module_modes: tuple[tuple[torch.nn.Module, bool], ...]

    @classmethod
    def capture(cls, actor: Any) -> "AuditState":
        module = actor.actor_module
        parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
        gradients = tuple(
            parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
            for parameter in parameters
        )
        buffers = tuple(
            (buffer, buffer.detach().cpu().clone())
            for buffer in module.buffers()
        )
        modes = tuple((child, child.training) for child in module.modules())
        return cls(
            rng=RngState.capture(),
            parameters=parameters,
            gradients=gradients,
            buffers=buffers,
            module_modes=modes,
        )

    def clear_gradients(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def restore_runtime(self) -> None:
        self.rng.restore()
        for module, training in self.module_modes:
            module.training = training
        for buffer, snapshot in self.buffers:
            buffer.copy_(snapshot.to(device=buffer.device, dtype=buffer.dtype))

    def restore(self) -> None:
        self.restore_runtime()
        for parameter, gradient in zip(self.parameters, self.gradients, strict=True):
            parameter.grad = (
                gradient.to(device=parameter.device, dtype=gradient.dtype).clone()
                if gradient is not None
                else None
            )
