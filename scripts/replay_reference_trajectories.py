#!/usr/bin/env python
"""Génère des trajectoires τ²-bench **réussies** en rejouant les actions de référence.

Débloque `scripts/validate_goal_signal.py` (gate étape 4).

⚠️ REQUALIFICATION (2026-07-15) — la JUSTIFICATION historique de ce script était fausse ; le
script, lui, reste valide. On lisait ici : « Qwen-32B nu échoue toutes les tâches retail
(baseline 0/8) ⇒ 0 trajectoire résolue ». Ce 0/8 ne mesurait PAS Qwen : le juge des
NL-assertions n'était pas câblé (défaut τ² = `gpt-4.1` → 404 → composante NL = 0), et 112/114
tâches retail portent une `NL_ASSERTION` dans `reward_basis` ⇒ `reward = db × 0 = 0` par
construction, quel qu'ait été le comportement de l'agent. **On ne sait toujours pas si Qwen nu
échoue sur retail** — la question est ouverte (cf. BENCHMARKS.md, bloc ⚠️ des lignes retail).
Ce que ça ne change pas : le corpus produit ici reste du VRAI (états τ² réellement traversés,
`db_reward` vérifié par l'évaluateur officiel), et rejouer la trajectoire experte reste la
bonne façon d'obtenir des positifs propres — même si Qwen s'avérait capable d'en résoudre.

Or τ² fournit, pour chaque tâche, la **trajectoire experte**
(`evaluation_criteria.actions`, toutes `requestor=assistant`).
En la rejouant contre l'environnement du domaine, on atteint par construction l'état-DB cible
(`db_reward == 1.0`, vérifié par le vrai évaluateur τ²) → autant de **positifs** propres.

C'est du VRAI (pas le mock synthetic disqualifié par token-leak) : ce sont les états τ²
réellement traversés par la trajectoire experte, mêmes textes de résultats d'outils que ceux
que la politique voit en observation (`Tau2Env.step` → `Observation.text`).

Positifs   : rejeu COMPLET des actions de référence → `success=True` (db_reward=1.0).
Négatifs   : rejeu TRONQUÉ (dernière action retirée) → `success=False` **si** db_reward<1.0
             (contrôle : même format, mêmes états, mais l'action qui « boucle » le but manque).
             Miroir contrôlé des échecs de Qwen, sans dépendre du serveur LLM (arrêté).

`goal` = l'instruction GÉNÉRIQUE non-solo (identique à celle que persiste le runner via
`Tau2Env.goal()`) — AUCUNE fuite du `user_scenario` (cf. `_build_goal`). Le score reste donc un
test HONNÊTE du signal goal-relative tel que morpheus l'utilise vraiment en retail.

Sortie : JSONL `{goal, success, states:[str,...], task, n_actions, reward, kind}` consommable par
`validate_goal_signal.py --trajectories <sortie> --checkpoint <jepa.pt>` (H1/H2 ; H3 = N/A sans
annotations manuelles ERREUR/NOUVEAUTÉ).

Exemple :
  python scripts/replay_reference_trajectories.py --domain retail --out data/tau2_replay/retail.jsonl
  python scripts/validate_goal_signal.py --trajectories data/tau2_replay/retail.jsonl \
         --checkpoint checkpoints/jepa_apigen/jepa.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# le backbone sentence-transformer est en cache local (cf. TODO Journal §4) : rester hors-ligne
# pour ne pas bloquer sur un accès réseau HF (le validateur charge le JEPA, pas ce script).
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _quiet_tau2() -> None:
    """τ² logue chaque réponse d'outil en DEBUG via loguru → on coupe (comme l'adaptateur)."""
    try:
        from loguru import logger

        logger.disable("tau2")
    except Exception:
        pass


def _replay(env_constructor, task, actions, evaluator, solo: bool = False):
    """Rejoue `actions` contre un env frais, renvoie (states, acts, reward, n_ok, had_error).

    `states[t]` = texte du résultat de la t-ième action (l'observation que la politique verrait),
    miroir de `TraceStep.real_state`. `reward` = db_reward de l'évaluateur τ² OFFICIEL sur la
    trajectoire construite (fresh predicted env vs gold env qui rejoue les actions de référence).
    `solo` : domaines à tickets (telecom) → le scoring DB officiel exige solo_mode=True (sinon le
    rejeu n'atteint pas l'état-but et les positifs sont ignorés)."""
    from tau2.data_model.message import AssistantMessage, ToolCall

    env = env_constructor(solo_mode=solo)
    trajectory = []
    states: list[str] = []
    acts: list[str] = []            # texte de l'action qui a produit chaque état (nom + args)
    n_ok = 0
    had_error = False
    for i, a in enumerate(actions):
        tc = ToolCall(id=f"c{i}", name=a.name, arguments=a.arguments or {}, requestor=a.requestor)
        tm = env.get_response(tc)  # exécute en direct, renvoie un ToolMessage (résultat)
        trajectory.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
        trajectory.append(tm)
        content = tm.content if isinstance(tm.content, str) else json.dumps(tm.content, default=str)
        states.append(content)
        acts.append(f"{a.name}({json.dumps(a.arguments or {}, ensure_ascii=False, default=str)})")
        if getattr(tm, "error", False):
            had_error = True
        else:
            n_ok += 1

    reward = None
    try:
        ri = evaluator.calculate_reward(env_constructor, task, list(trajectory), solo_mode=solo)
        reward = float(ri.reward)
    except Exception as e:  # pragma: no cover — robustesse par tâche
        print(f"    ⚠️  évaluateur KO : {e}", file=sys.stderr)
    return states, acts, reward, n_ok, had_error


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rejoue les actions de référence τ² → trajectoires réussies.")
    ap.add_argument("--domain", default="retail", help="domaine τ² (retail, airline, telecom, mock)")
    ap.add_argument("--out", default="data/tau2_replay/retail.jsonl", help="JSONL de sortie")
    ap.add_argument("--limit", type=int, default=0, help="limiter le nombre de tâches (0 = toutes)")
    ap.add_argument("--min-len", type=int, default=3,
                    help="longueur mini d'une trajectoire conservée (défaut 3 = MIN_LEN du validateur)")
    ap.add_argument("--solo", action="store_true",
                    help="domaines à tickets (telecom) : solo_mode=True pour le rejeu ET le scoring "
                         "DB, et but = ticket PAR TÂCHE (brief légitime, non-fuyant) → buts distincts "
                         "⇒ InfoNCE inter-buts exploitable (contrairement au but générique non-solo).")
    ap.add_argument("--no-negatives", action="store_true",
                    help="ne pas générer les négatifs tronqués (positifs seulement)")
    ap.add_argument("--neg-fracs", default="0.4,0.65,0.9",
                    help="fractions de troncature des négatifs (fraction d'actions CONSERVÉES). "
                         "Plusieurs points → négatifs de difficulté variée (échecs précoces ET "
                         "tardifs) → classe échec au signal plus tranché pour H2.")
    args = ap.parse_args(argv)

    _quiet_tau2()
    try:
        from tau2.evaluator.evaluator_env import EnvironmentEvaluator
        from tau2.registry import registry
    except ImportError as e:
        print(f"❌ τ²-bench non installé : {e}", file=sys.stderr)
        return 2

    from morpheus.envs.tau2_adapter import _build_goal  # instruction générique non-solo (non-fuyante)

    env_constructor = registry.get_env_constructor(args.domain)
    tasks = registry.get_tasks_loader(args.domain)()
    if args.limit:
        tasks = tasks[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pos = n_pos_usable = n_neg = n_skip_reward = 0
    # non-solo : but générique unique. solo : but = ticket PAR TÂCHE (calculé dans la boucle).
    generic_goal = _build_goal(None, solo=False, domain=args.domain)

    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            actions = list(task.evaluation_criteria.actions or [])
            if not actions:
                continue
            goal = _build_goal(task, solo=True, domain=args.domain) if args.solo else generic_goal

            # --- POSITIF : rejeu complet → doit atteindre l'état-but (db_reward == 1.0) ---
            states, acts, reward, n_ok, had_error = _replay(
                env_constructor, task, actions, EnvironmentEvaluator, solo=args.solo)
            success = reward is not None and reward >= 1.0 - 1e-9
            if not success:
                n_skip_reward += 1
                print(f"  [{task.id}] positif NON résolu (reward={reward}, err={had_error}) — ignoré",
                      file=sys.stderr)
            else:
                n_pos += 1
                rec = {"goal": goal, "success": True, "states": states, "actions": acts,
                       "task": task.id, "n_actions": len(actions), "reward": reward,
                       "kind": "replay_full"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if len(states) >= args.min_len:
                    n_pos_usable += 1

            # --- NÉGATIFS : rejeux tronqués à PLUSIEURS points → échecs de difficulté variée ---
            # Une seule troncature « dernière action » progresse presque comme un succès (H2
            # marginal). En coupant aussi PLUS TÔT, la trajectoire s'arrête loin de la résolution :
            # jamais le saut final → pente plus faible ⇒ meilleure séparation succès/échec.
            if not args.no_negatives:
                L = len(actions)
                fracs = [float(x) for x in args.neg_fracs.split(",") if x.strip()]
                cuts = sorted({max(args.min_len, min(L - 1, round(fr * L))) for fr in fracs})
                cuts = [k for k in cuts if args.min_len <= k <= L - 1]   # ≥ min_len, ≥1 action retirée
                for k in cuts:
                    t_states, t_acts, t_reward, _, _ = _replay(
                        env_constructor, task, actions[:k], EnvironmentEvaluator, solo=args.solo
                    )
                    # ne garder que les VRAIS échecs (n'atteint pas l'état-but)
                    if t_reward is not None and t_reward < 1.0 - 1e-9 and len(t_states) >= args.min_len:
                        n_neg += 1
                        rec = {"goal": goal, "success": False, "states": t_states, "actions": t_acts,
                               "task": task.id, "n_actions": k, "reward": t_reward,
                               "kind": f"replay_trunc_{k}"}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ {out_path}")
    print(f"   positifs (résolus)         : {n_pos}  (dont exploitables ≥{args.min_len} états : {n_pos_usable})")
    print(f"   négatifs (tronqués, échec) : {n_neg}")
    if n_skip_reward:
        print(f"   ⚠️  positifs ignorés (reward<1 au rejeu) : {n_skip_reward}")
    print(f"\n   → python scripts/validate_goal_signal.py --trajectories {out_path} \\")
    print(f"          --checkpoint checkpoints/jepa_apigen/jepa.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
