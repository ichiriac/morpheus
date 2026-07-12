"""Configuration de l'orchestrateur, chargée depuis un YAML (voir configs/phase1.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """Un backend LLM. `kind` ∈ {stub, openai, anthropic}."""

    kind: str = "stub"
    model: str = "stub"
    base_url: str | None = None          # ex: http://localhost:8000/v1 (vLLM/llama.cpp)
    api_key_env: str | None = None       # nom de la variable d'env contenant la clé
    temperature: float = 0.7
    max_tokens: int = 1024
    # passé tel quel au backend (ex. Qwen3 sans thinking :
    #   {"chat_template_kwargs": {"enable_thinking": false}})
    extra_body: dict = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Paramètres de la boucle fermée (MPC à horizon glissant)."""

    k_candidates: int = 4                 # nb d'actions candidates proposées par tour (K)
    horizon: int = 3                      # profondeur du lookahead latent/texte (H)
    max_turns: int = 12                   # T_max
    surprise_threshold: float = 0.5       # seuil de déclenchement du routeur de surprise
    use_world_model: bool = True          # False = baseline ReAct nue (Phase 0)


@dataclass
class EvalConfig:
    env: str = "mock"                     # mock | tau2
    domain: str = "retail"                # τ²-bench : retail | airline | telecom
    tasks: int = 20
    turn_buckets: list[int] = field(default_factory=lambda: [4, 8, 12])
    seed: int = 0
    out_dir: str = "runs"


@dataclass
class Config:
    policy: LLMConfig = field(default_factory=LLMConfig)
    world_model: LLMConfig = field(default_factory=LLMConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            policy=LLMConfig(**(data.get("policy") or {})),
            world_model=LLMConfig(**(data.get("world_model") or {})),
            orchestrator=OrchestratorConfig(**(data.get("orchestrator") or {})),
            eval=EvalConfig(**(data.get("eval") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
