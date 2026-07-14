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
    # nb de rollouts LLM concurrents (les K rollouts sont indépendants). >1 => vLLM batche
    # les requêtes en vol au lieu de les traiter en série. 1 = séquentiel (déterministe/CI).
    concurrency: int = 1
    # --- RAG gated par la surprise (Phase 3) ---
    use_rag: bool = False                 # True = récupère la KB (policy) quand δ dépasse le seuil
    rag_top_k: int = 3                    # nb de règles de policy récupérées par surprise
    # Mémoire épisodique de faits atomiques (LWM-Planner) : accumule les faits des observations
    # RÉELLES et les récupère sur surprise. Sort du régime redondant du RAG-sur-policy (la policy
    # est déjà dans le system_context ; les faits observés, non). Ablatable indépendamment.
    use_memory: bool = False
    memory_top_k: int = 3
    # Sonde « réductibilité » (signal 4 du tableau specs/01) : sur surprise, demande au
    # world-model LLM si l'écart prédit↔réel s'explique sans faute (+1 appel LLM/surprise).
    # Sans objet avec JepaWorldModel (latent non verbalisable → signal None). Off par défaut :
    # zéro surcoût, zéro changement de comportement.
    use_reducibility: bool = False


@dataclass
class EvalConfig:
    env: str = "mock"                     # mock | tau2
    domain: str = "retail"                # τ²-bench : retail | airline | telecom
    tasks: int = 20
    turn_buckets: list[int] = field(default_factory=lambda: [4, 8, 12])
    seed: int = 0
    out_dir: str = "runs"
    # mock « planning » : l'observation ne révèle plus l'étape suivante → charge de planification
    # croissante avec la longueur (régime où le lookahead du world-model peut départager les K).
    mock_hard: bool = False
    # --- τ²-bench (env: tau2) ---
    # solo=True : agent seul (DummyUser, hors-ligne) ; n'accepte QUE les tâches avec un
    # `ticket` (telecom/mock). solo=False : user simulé par LLM (retail/airline) → renseigner
    # tau2_user_* pour pointer le simulateur sur un endpoint (ex. le même vLLM Qwen).
    tau2_solo: bool = False
    tau2_max_steps: int = 30              # garde-fou interne τ² (≥ 2×max_turns conseillé)
    tau2_split: str | None = None         # nom de split de tâches (optionnel)
    tau2_user_llm: str | None = None      # modèle litellm du user-sim (ex. openai/Qwen/Qwen3-32B-AWQ)
    tau2_user_base_url: str | None = None # base_url du user-sim (ex. http://localhost:8000/v1)
    tau2_user_api_key_env: str | None = None
    # Juge LLM des NL-assertions (calcul du reward retail/airline : 112/114 tâches retail ont
    # NL_ASSERTION dans leur reward_basis → sans ce juge, reward = db × 0 = 0). Défaut τ² =
    # gpt-4.1 (clé OpenAI). Renseigner pour le pointer sur le vLLM local (ex. mêmes valeurs que
    # user-sim). ⚠️ Qwen jugeant Qwen est méthodologiquement faible : mesure indicative.
    tau2_judge_llm: str | None = None     # ex. openai/Qwen/Qwen3-32B-AWQ
    tau2_judge_base_url: str | None = None
    tau2_judge_api_key_env: str | None = None
    # --- KB / RAG (orchestrator.use_rag) ---
    # Source du référentiel de vérité : les policy.md de τ². Si kb_policy_path est None, on
    # dérive `<tau2_data_dir|$TAU2_DATA_DIR|./tau2-bench/data>/tau2/domains/<domain>/policy.md`.
    kb_policy_path: str | None = None
    tau2_data_dir: str | None = None


@dataclass
class JepaWMConfig:
    """World-model LATENT (JEPA). OPTIONNEL : si `enabled=False` (défaut), le runner garde le
    LLM-as-world-model et torch n'est jamais importé. `enabled=True` charge `checkpoint`
    (jepa.pt) et branche `JepaWorldModel` à la place — loop.py inchangé (même contrat)."""

    enabled: bool = False
    checkpoint: str = "checkpoints/jepa/jepa.pt"
    device: str = "auto"                  # auto | cpu | cuda


@dataclass
class Config:
    policy: LLMConfig = field(default_factory=LLMConfig)
    world_model: LLMConfig = field(default_factory=LLMConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    jepa_wm: JepaWMConfig = field(default_factory=JepaWMConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            policy=LLMConfig(**(data.get("policy") or {})),
            world_model=LLMConfig(**(data.get("world_model") or {})),
            orchestrator=OrchestratorConfig(**(data.get("orchestrator") or {})),
            eval=EvalConfig(**(data.get("eval") or {})),
            jepa_wm=JepaWMConfig(**(data.get("jepa_wm") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
