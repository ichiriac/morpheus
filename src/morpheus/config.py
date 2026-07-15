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
    # Seuil de déclenchement du routeur de surprise. `None` (défaut) = **le world-model décide**
    # (`WorldModel.surprise_threshold`) : δ n'est PAS la même grandeur selon le WM — Jaccard sur
    # du texte pour le LLM (échelle [0,1] réellement parcourue), (1−cos)/2 dans un latent pour le
    # JEPA (deux vecteurs sans rapport valent ≈0.26 ⇒ δ plafonne vers ≈0.37). Un seuil unique
    # porté par l'orchestrateur se transmet d'un WM à l'autre et se retrouve hors de l'échelle
    # atteignable — c'est ce qui rendait le routeur JEPA muet (0.5 exigeait « plus faux qu'un
    # tirage au hasard »). Une valeur explicite ici reste possible : elle PRIME sur le world-model.
    surprise_threshold: float | None = None
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
    # --- Routeur de surprise APPRIS (Phase 4) ---
    # None = règle Phase 1 (2 signaux, `SurpriseRouter`). Un chemin vers `router.json`
    # (`morpheus train-router`) substitue le routeur appris : même contrat `route(signals)`,
    # numpy pur, pas de torch. Le RÉGIME DE SONDAGE doit correspondre à celui de l'entraînement
    # (`use_rag`/`use_memory`/…) — sinon les features non sondées sont épinglées à 0 et le logit
    # dérive silencieusement ; le runner refuse le run (cf. RouterModel.regime_drift).
    router_checkpoint: str | None = None


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
    # Manifeste de banc : chemin d'un JSON `{"task_ids": ["0", "3", …]}` (cf.
    # scripts/build_nojudge_bench.py). Sélectionne EXACTEMENT ces tâches, dans l'ordre du
    # manifeste — au lieu du défaut « les N premières ». C'est la DÉFINITION d'un banc : tous les
    # bras d'une comparaison (baseline / world-model / Sonnet / Qwen-natif) doivent tourner sur la
    # même liste, sinon les écarts ne sont pas interprétables. Un id absent du domaine est une
    # ERREUR (échouer ici, pas mesurer un banc silencieusement amputé). `tasks` tronque encore la
    # liste si elle est plus courte (pratique pour un smoke sur le même banc).
    tau2_task_ids_file: str | None = None
    tau2_user_llm: str | None = None      # modèle litellm du user-sim (ex. openai/Qwen/Qwen3-32B-AWQ)
    tau2_user_base_url: str | None = None # base_url du user-sim (ex. http://localhost:8000/v1)
    tau2_user_api_key_env: str | None = None
    # Juge LLM des NL-assertions. Défaut τ² = gpt-4.1 (clé OpenAI) → 404 sur un pod sans clé.
    # Renseigner pour le pointer sur le vLLM local (ex. mêmes valeurs que user-sim).
    #
    # ⚠️ CORRIGÉ 2026-07-15 — ce commentaire portait « 112/114 tâches retail ont NL_ASSERTION dans
    # reward_basis → sans ce juge, reward = db × 0 = 0 ». L'inférence est FAUSSE et elle a essaimé
    # dans 4 autres fichiers avant d'atteindre un commit. Ce qui déclenche le juge n'est PAS
    # `reward_basis` mais `nl_assertions` NON VIDE (tau2/evaluator/evaluator_nl_assertions.py) :
    #   · `NL_ASSERTION` dans reward_basis .... 112/114  ← ne déclenche rien
    #   · `nl_assertions` non vide ............  40/114  ← seules celles-ci appellent le juge
    #   · `nl_assertions` vide ................  74/114  ← `if not nl_assertions: return 1.0`,
    #     sortie anticipée AVANT tout appel LLM ⇒ NL=1.0 vacu, aucun 404 possible, reward = db.
    # Donc sans juge : 40 tâches cassées, 74 tâches parfaitement mesurées. Pas « par construction ».
    #
    # ⚠️ Qwen jugeant Qwen est faible ET, mesuré : le juge ne discrimine pas. Seul appel réel du
    # smoke `retail_attrib` (BENCHMARKS.md) : assertion « there are 10 t-shirt options available »
    # → verdict MET alors que l'agent en a listé 9 sans jamais dire « 10 ». Faux positif 1/1.
    # ⇒ Lire le `DB` seul. Les 74 tâches à nl_assertions vide forment un banc PROPRE (reward = db,
    # protocole τ² officiel, aucun juge dans la boucle).
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
