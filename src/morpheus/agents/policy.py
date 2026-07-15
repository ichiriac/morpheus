"""Politique = Qwen. Propose K actions candidates au tour courant (étape PROPOSER)."""

from __future__ import annotations

import json
import re

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
_TRANSCRIPT_TURNS = 8
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
        return (
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
        if not actions:  # filet de sécurité : au moins une action valide
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
