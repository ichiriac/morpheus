# Annotations manuelles — ERREUR vs NOUVEAUTÉ (H3)

Vérité-terrain pour l'hypothèse **H3** de `scripts/validate_goal_signal.py` : est-ce que
`score_after < score_before` (signal dérivé du world-model) sépare les pas où l'agent **a fauté**
(ERREUR) de ceux où **le monde s'est révélé plus riche que le plan** (NOUVEAUTÉ) ? Ces labels
sont indépendants du routeur automatique (`agents/surprise.py`) — sinon le test serait circulaire.

## Contenu (lot POST-FIX ARGS, 2026-07-13)

- `tau2_error_novelty.jsonl` — **109 annotations** (30 ERREUR / 79 NOUVEAUTÉ) sur 9 trajectoires
  τ² **post-correction du bug ARGS** (commit `31a3390`). Une ligne par pas :
  `{episode, turn, label, rationale, chosen, evidence, annotator, annotated_on}`.
- `trajectories/<run>/episodes.jsonl` — les trajectoires annotées, **versionnées** ici (le lot v1
  pointait sur des runs gitignorés = irreproductible). Clé de jointure `episode = "<run>#<task>"`.

## Rubrique (d'après specs/01 §« routeur de surprise »)

| Label | `rationale` | Définition |
|---|---|---|
| **ERROR** | `tool_error` | l'outil renvoie une erreur explicite (`Error`, `not found`, `failed`) |
| **ERROR** | `loop_no_progress` | même outil que le pas précédent (qui n'avait pas erré) → l'agent répète sans avancer |
| **ERROR** | `coherent_but_wrong` | appel **réussi** mais factuellement à côté, révélé plus tard (le *cohérent-mais-faux*) |
| **NOVELTY** | `user_new_info` | réponse utilisateur légitime (nouvelle contrainte, identité, préférence) |
| **NOVELTY** | `tool_success` | appel réussi porteur d'information |

Répartition : `user_new_info` 44 · `tool_success` 35 · `tool_error` 19 · `loop_no_progress` 10 ·
`coherent_but_wrong` 1.

**Pourquoi ce lot est meilleur que le v1** : **11 des 30 ERREUR sont des pas où l'outil a RÉUSSI**
(boucles + cohérent-mais-faux). Le label n'est donc **plus réductible au signal `tool_error`** (un
input du routeur) → H3 teste enfin le pouvoir discriminant *propre* du score, pas juste « l'outil
a planté ». Cas emblématique : `retail_postfix#1` t15 — `modify_pending_order_address` réussit sur
la **mauvaise commande**, révélé quand l'utilisateur corrige à t16. C'est exactement la classe
d'erreurs que JEPA seul ne peut pas voir (cf. specs/00 §« le rôle de la connaissance »).

## Statut d'exécution de H3 : encore en attente (mais plus sur l'annotation)

L'annotation n'est plus le blocage. Restent deux prérequis côté *scores* :

1. **JEPA sémantique** requis pour calculer `score_to_goal` (l'actuel checkpoint est hashing/
   synthetic → bruit). Tant qu'il manque, H3 = `N/A` (le harnais le dit).
2. **⚠️ Goal générique en retail non-solo** : pour ne pas fuiter le besoin utilisateur, le `goal`
   persisté des tâches retail est l'instruction GÉNÉRIQUE (cf. `tau2_adapter._NONSOLO_GOAL`), pas le
   vrai objectif de la tâche. `score_to_goal(goal_générique, état)` n'y mesure donc PAS une
   progression réelle → la validation goal-relative (H1/H2/H3) est significative surtout sur les
   trajectoires **solo** (telecom : le `ticket` EST l'objectif, légitimement donné à l'agent).
   Les labels ERREUR/NOUVEAUTÉ restent corrects partout ; c'est le *scoring* qui est limité en retail.

Annotateur = **modèle** (marqué tel quel), rubrique explicite après lecture des 109 pas —
**relecture humaine recommandée**, surtout sur `loop_no_progress` et `coherent_but_wrong`.

## Usage

```bash
python scripts/validate_goal_signal.py \
  --episodes data/annotations/trajectories/retail_postfix/episodes.jsonl \
             data/annotations/trajectories/telecom_solo_postfix/episodes.jsonl \
  --labels  data/annotations/tau2_error_novelty.jsonl \
  --checkpoint checkpoints/jepa/jepa.pt      # JEPA SÉMANTIQUE requis pour un H3 réel
```
