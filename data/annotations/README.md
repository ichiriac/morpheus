# Annotations manuelles — ERREUR vs NOUVEAUTÉ (H3)

Vérité-terrain pour l'hypothèse **H3** de `scripts/validate_goal_signal.py` : est-ce que
`score_after < score_before` (signal dérivé du world-model) sépare les pas où l'agent **a fauté**
(ERREUR) de ceux où **le monde s'est révélé plus riche que le plan** (NOUVEAUTÉ) ?
Ces labels sont indépendants du routeur automatique (`agents/surprise.py`) — sinon le test serait
circulaire.

## Rubrique (d'après specs/01 §« routeur de surprise »)

| Label | Définition | Signaux concrets |
|---|---|---|
| **ERROR** | « j'ai fauté » — pas à corriger | l'outil renvoie une **erreur explicite** (exception, `unexpected keyword argument`, `not found`, 4xx/5xx) ; ou l'état s'éloigne clairement du but par la faute de l'agent |
| **NOVELTY** | « le monde est plus riche que mon plan » — à assimiler | appel d'outil **réussi** renvoyant une info valide inattendue ; ou **nouvelle information légitime de l'utilisateur** (nouvelle contrainte, préférence, identité) |

`rationale` ∈ `tool_error` · `tool_not_found` · `tool_success_unexpected` · `user_new_info`.

## Fichiers

- `tau2_error_novelty.jsonl` — une annotation par ligne :
  `{episode, turn, label, rationale, chosen, evidence, annotator, annotated_on}`.
  Clé de jointure `episode = "<nom_du_run>#<task>"` (ex. `qwen_tau2_retail_smoke#0`), à recouper
  avec `runs/<run>/episodes.jsonl` via `--episodes` / `--labels`.

## ⚠️ Statut : PROVISOIRE — ne PAS conclure H3 dessus

Ce lot annote les **6 seules trajectoires τ² réelles disponibles au 2026-07-13**, qui échouent
**toutes** sur le bug ARGS placeholder (`_SYS` : `{"clef":"valeur"}` → appels malformés). Conséquences :

1. **Dégénérescence** : ERREUR ≈ « l'outil a planté » (déjà un signal du routeur) et NOUVEAUTÉ ≈
   « réponse utilisateur / appel réussi ». Le label corrèle donc trop avec `tool_error` → H3 ne
   testerait pas le pouvoir discriminant *propre* du score latent.
2. **Pas de scores** : `score_to_goal` exige un JEPA **sémantique** + le `goal` (les 6 runs sont
   d'avant la persistance du goal). Tant que ça manque, H3 reste `N/A`.

**À refaire** sur des runs τ² post-correction ARGS (vraies réussites + échecs variés), idéalement
avec **relecture humaine** de ces labels (annotateur = modèle, marqué comme tel). Ce lot valide le
**pipeline** (format, jointure, gitignore), pas encore le signal.

## Usage

```bash
python scripts/validate_goal_signal.py \
  --episodes runs/qwen_tau2_retail_smoke/episodes.jsonl runs/qwen_tau2_retail_check/episodes.jsonl \
             runs/qwen_tau2_telecom_solo/episodes.jsonl \
  --labels  data/annotations/tau2_error_novelty.jsonl \
  --checkpoint checkpoints/jepa/jepa.pt      # requis pour calculer les scores → H3 réel
```
