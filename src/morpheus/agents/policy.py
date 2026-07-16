"""Politique = Qwen. Propose K actions candidates au tour courant (étape PROPOSER)."""

from __future__ import annotations

import json
import re
import sys

from .. import prompt_tags as T
from ..llm.base import LLMClient, system, user
from ..orchestrator.types import Action, State
from ..text import snap_to_whitelist, strip_reasoning

_SYS = (
    "Tu es la politique d'un agent outillé. À partir de l'objectif et de l'état courant, "
    "tu proposes plusieurs actions candidates PLAUSIBLES et DISTINCTES pour le prochain "
    "pas — pas le plan entier, juste le prochain coup.\n"
    "Contraintes STRICTES :\n"
    "- N'utilise QUE des outils de la liste fournie (noms exacts).\n"
    "- Une action par ligne, format exact : "
    "`ACTION: <nom_outil> | ARGS: {\"arg\": valeur}`.\n"
    "- ARGS = les VRAIS arguments de l'outil (vrais noms, vraies valeurs tirées de l'objectif "
    "et de l'état), objet JSON — pas de placeholder.\n"
    "- Si l'outil `respond_to_user` est proposé, c'est pour PARLER à l'utilisateur (demander une "
    "info manquante, confirmer) : `ACTION: respond_to_user | ARGS: {\"text\": \"...\"}`.\n"
    "- Pas de texte autour, pas de numérotation, pas de commentaire."
)

# Fenêtre de mémoire de la politique (scratchpad ReAct) : nb de couples action→résultat gardés
# et longueur max de chaque résultat (borne le prompt tout en gardant l'info utile).
#
# 40 (2026-07-17, MESURÉ) — le 8 précédent était une prudence non chiffrée, et il coûtait cher :
# **92 % des épisodes (68/74) de `retail74_baseline_run2` dépassaient 8 tours**, donc dans 92 %
# des cas l'agent avait déjà oublié le début de sa propre trajectoire. Une fenêtre de 8 dans un
# budget de 16 est une mémoire qui ne fonctionne quasiment jamais.
# Le coût de la lever est NUL : les observations τ²-retail font **610 car. en moyenne** (médiane
# 287, p95 1961 ; n=1015 observations réelles) ⇒ ~149 tokens/tour après cap. Budget : 40 tours ≈
# 6k tokens de scratchpad + ~6k de système/outils ≈ 12k, très en dessous des 32768 de MAX_LEN
# (même 100 tours tiendraient : ≈21k). Aligné sur `max_turns` — garder les deux cohérents.
_TRANSCRIPT_TURNS = 40
# 2500 = premier palier au-dessus du p95 MESURÉ des résultats d'outils τ²-retail (p95 = 2164 sur
# 1357 payloads : data/tau2_replay/retail.jsonl + trace du smoke `retail_attrib`). Couvre 96.2%
# des résultats ; coût ≈ 8×2500/4 ≈ 5k tokens, soit 15% d'un contexte 32k — négligeable.
#
# Le 600 précédent ne couvrait que 38.6% : il TRONQUAIT 62.6% des résultats (459 car. perdus en
# moyenne). Ce n'était pas un cas limite mais le régime normal. Défaut observé dans `retail_attrib`
# (BENCHMARKS.md) : `list_all_product_types` rend la table nom→ID en 1478 car. ; l'entrée utile
# ("Mechanical Keyboard": "1656367028") est à l'offset 812, donc ÉVINCÉE du scratchpad. L'agent
# rappelait ensuite `get_product_details(product_id='Mechanical Keyboard')` en boucle, sans plus
# aucun moyen de retrouver l'ID : le rattrapage devenait structurellement impossible.
# ⚠️ Ne PAS confondre avec `_TRANSCRIPT_TURNS`, hors de cause ici (t10→t13 = 3 tours, cap à 8).
# ⚠️ Ce cap ne s'applique qu'au RENDU du prompt : `transcript` garde les textes entiers, donc le
# signal `familiarity` du routeur (loop.py) n'est pas affecté. En revanche le prompt de la
# politique CHANGE ⇒ les runs d'avant 2026-07-15 ne sont pas comparables à ceux d'après.
_TRANSCRIPT_CHARS = 2500


class Policy:
    def __init__(self, llm: LLMClient, k: int = 4) -> None:
        self.llm = llm
        self.k = k
        self.n_parse_fallback = 0     # cf. `propose` : compte les replis « aucune action parsée »

    def build_prompt(self, state: State, tools: list[str],
                     transcript: list[tuple[str, str]] | None = None,
                     facts: list[str] | None = None, route: str | None = None) -> str:
        # `transcript` = derniers couples (action → résultat d'outil/message). Sans lui, la
        # politique n'aurait que la DERNIÈRE observation et oublierait les résultats passés
        # (amnésie fatale en tool-use multi-tours). Rendu seulement au vrai PROPOSER.
        if transcript:
            lines = "\n".join(
                f"- {act} → { ' '.join(res.split())[:_TRANSCRIPT_CHARS] }"
                for act, res in transcript[-_TRANSCRIPT_TURNS:]
            )
            hist = f"[HISTORIQUE ACTION→RÉSULTAT]\n{lines}\n[/HISTORIQUE]\n"
        else:
            hist = f"Historique récent : {state.history[-5:]}\n"
        # `facts` = règles de la KB récupérées suite à une SURPRISE au tour précédent (RAG gated).
        # ERROR → l'action a fauté : corriger. NOVELTY → le monde est plus riche : assimiler.
        kb = ""
        if facts:
            consigne = ("Ton coup précédent a FAUTÉ au regard de ces règles — propose une action "
                        "CONFORME qui corrige."
                        if route == "ERROR" else
                        "Le monde s'est révélé plus riche que ton plan — ASSIMILE ces règles et "
                        "adapte ton prochain coup.")
            kb = ("[CONNAISSANCE RÉCUPÉRÉE — suite à surprise, à respecter]\n"
                  + "\n".join(f"- {f}" for f in facts)
                  + f"\n{consigne}\n[/CONNAISSANCE]\n")
        # Où en est-on, et combien reste-t-il ? Absent jusqu'au 2026-07-17 : `state.turn` existait
        # mais n'était JAMAIS rendu — la politique ne pouvait pas se rythmer sur ce qu'elle ne
        # voyait pas, alors que 59 % des épisodes finissaient au plafond.
        # Formulation STRICTEMENT FACTUELLE, à dessein : le mode d'échec dominant est l'écriture
        # prématurée (~50 % des échecs), donc une consigne du type « conclus dès que possible »
        # l'aggraverait. On énonce la contrainte, on ne prescrit pas de stratégie.
        # ⚠️ L'effet est une EXPÉRIENCE, pas un correctif : la conscience du budget peut réduire la
        # flânerie (bon) OU induire une précipitation (mauvais). À lire dans la re-mesure.
        budget = ""
        if state.turn > 0:
            budget = T.block(
                T.BUDGET,
                f"Tour {state.turn}/{state.max_turns} — il reste "
                f"{max(0, state.max_turns - state.turn)} tours ; au-delà l'épisode s'arrête, "
                f"résolu ou non."
                if state.max_turns > 0 else f"Tour {state.turn}.",
            ) + "\n"
        return (
            f"{budget}"
            f"{T.block(T.GOAL, state.goal)}\n"
            f"{T.block(T.POLICY_STATE, state.text)}\n"
            f"{hist}{kb}"
            f"{T.block(T.CANDIDATE_TOOLS, ', '.join(tools))}\n"
            f"Propose jusqu'à {self.k} actions candidates distinctes."
        )

    def propose(self, state: State, tools: list[str],
                system_context: str | None = None,
                transcript: list[tuple[str, str]] | None = None,
                facts: list[str] | None = None, route: str | None = None) -> list[Action]:
        # `system_context` (policy domaine + signatures), `transcript` (mémoire action→résultat) et
        # `facts` (KB récupérée sur surprise) ne sont passés QU'au vrai pas PROPOSER. Les rollouts
        # imaginés du world-model appellent propose() sans eux → prompts K·H bornés, WM hors-RAG.
        sys = _SYS
        if system_context:
            sys = f"{_SYS}\n\n[POLICY DU DOMAINE — à respecter strictement]\n{system_context}"
        raw = self.llm.complete(
            [system(sys), user(self.build_prompt(state, tools, transcript, facts, route))]
        )
        actions = _parse_actions(raw, tools)
        if not actions:
            # Filet de sécurité : au moins une action valide. ⚠️ SILENCIEUX jusqu'au 2026-07-17 —
            # or un repli sur `tools[0]()` SANS arguments erre presque toujours : c'est un tour
            # perdu, indiscernable d'une décision dans la trace. Avec `enable_thinking: true`, un
            # `max_tokens` trop court tronque le bloc <think> AVANT les lignes ACTION : plus aucune
            # action parsable, le filet se déclenche à CHAQUE tour, et un run de 9 h est empoisonné
            # sans qu'un seul message ne l'indique. On le rend BRUYANT — `grep PARSE_FALLBACK` sur
            # le log suffit à invalider un run avant de l'interpréter.
            self.n_parse_fallback += 1
            print(
                f"⚠️  PARSE_FALLBACK #{self.n_parse_fallback} : aucune action parsée dans "
                f"{len(raw)} car. de sortie brute (thinking tronqué ? <think> non fermé ?) → "
                f"repli sur {tools[0] if tools else 'noop'}() — vérifier `policy.max_tokens`.",
                file=sys.stderr, flush=True,
            )
            actions = [Action(tool=tools[0])] if tools else [Action(tool="noop")]
        return actions[: self.k]


def _parse_actions(raw: str, tools: list[str]) -> list[Action]:
    """Parse tolérant pour de vrais LLM : retire le raisonnement, lit les lignes ACTION,
    et, à défaut, repère les noms d'outils autorisés apparaissant dans le texte. Tout nom
    est ramené (snap) sur la whitelist ; les hallucinations non résolubles sont écartées."""
    text = strip_reasoning(raw)
    actions: list[Action] = []
    seen: set[str] = set()

    def add(tool_raw: str, args: dict) -> None:
        tool = snap_to_whitelist(tool_raw, tools)
        if tool is None:
            return
        key = f"{tool}:{sorted(args.items())}"
        if key in seen:
            return
        seen.add(key)
        actions.append(Action(tool=tool, args=args))

    # 1) format attendu : lignes `ACTION: <tool> | ARGS: {…}`
    for line in text.splitlines():
        m = re.search(r"ACTION:\s*([\w.\-]+)", line)
        if not m:
            continue
        args: dict = {}
        am = re.search(r"ARGS:\s*(\{.*\})", line)
        if am:
            try:
                args = json.loads(am.group(1))
            except json.JSONDecodeError:
                args = {}
        add(m.group(1), args)

    # 2) filet : si le modèle n'a pas suivi le format, repérer les outils cités dans l'ordre
    if not actions:
        for tok in re.findall(r"[\w.\-]+", text):
            if tok in tools:
                add(tok, {})

    return actions
