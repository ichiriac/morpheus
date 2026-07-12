# 05 — Entraînement du JEPA (Phase 2)

> World-model latent. On entraîne le **prédicteur `P`** à prévoir l'embedding de l'état
> résultant, avec un **encodeur `E_state` gelé** (option A de specs/01). Code : `src/morpheus/jepa/`.

## Ce que le JEPA apprend

Une transition = `(obs, action, next_obs)`. Le prédicteur apprend la **dynamique** de
l'environnement d'outils dans l'espace latent :

```
 s   = E_state(obs)          # embedding gelé (hashing | sentence-transformer | Qwen)
 s'  = E_state(next_obs)
 z   = proj(s)               # projection apprise
 a   = enc_action(E_state(action))
 ẑ'  = P(z, a)               # PRÉDICTION
 perte = 1 - cos(ẑ', proj(s').detach())  +  VICReg(anti-collapse)
```

Le stop-grad sur la cible + **VICReg** (variance + covariance) empêchent l'effondrement (tout
projeter sur un point). Voir `model.py`, `losses.py`.

## Séparation torch / torch-free (volontaire)

| Module | torch ? | Validable sans GPU |
|---|---|---|
| `jepa/data.py` (normalisation trajectoires) | non | **oui** (testé ici) |
| `jepa/encoders.py` (HashingEncoder, SentenceTransformer) | non | **oui** |
| `jepa/model.py`, `losses.py`, `train.py` | oui | sur RunPod |

Les tests torch sont guardés (`pytest.importorskip`) : 16 passent ici, 2 skippés (torch),
qui s'exécuteront sur le pod.

## Données : ce qui compte

Le prédicteur a besoin de l'**état résultant** → filtrer sur *« y a-t-il `next_obs` ? »*.
`data.py` normalise plusieurs schémas vers `Transition(obs, action, next_obs, reward, done)` :

- **`from_messages`** : conversations OpenAI (`tool_calls`, `role="tool"`) **et** ShareGPT
  (`from`/`value`, `function_call`/`observation`). Couvre **APIGen-MT-5k** et la plupart des
  datasets multi-tours (cf. specs/04 pour la liste HF).
- **`from_alfworld_steps`** : étapes `(observation, action)` → transitions consécutives.
- **`synthetic_transitions`** : jouet retail-lite, aucun téléchargement (smoke).

> ⚠️ Les noms de champs HF varient. **Toujours** vérifier d'abord :
> ```bash
> morpheus inspect-data --source hf:Salesforce/APIGen-MT-5k --limit 20
> ```
> Si l'aperçu est vide/faux, ajuster le mapping dans `from_messages` (comme le TODO tau2).

## Lancer un entraînement

```bash
# smoke (aucun download, CPU ou GPU) : valide toute la mécanique
morpheus train-jepa --config configs/jepa.yaml           # source=synthetic, encoder=hashing

# vrai run sur RunPod : éditer configs/jepa.yaml
#   source: hf:Salesforce/APIGen-MT-5k
#   encoder: sentence_transformer         # embeddings sémantiques
#   epochs: 50 ...
morpheus train-jepa --config configs/jepa.yaml
```

Sortie dans `checkpoints/jepa/` : `jepa.pt` (meilleur val), `history.json`, `config.json`.

## Ordre recommandé (rappel)

1. **Pré-entraîner `P`** sur APIGen-MT-5k (+ ALFWorld) — dynamique générique de tool-use.
2. **Fine-tuner** sur tes **propres rollouts τ²-bench** (Phase 1) : la seule donnée
   parfaitement on-distribution. → export via `to_jsonl`, puis `source: jsonl:<path>`.
3. Choisir `E_state` : commencer `sentence_transformer` (léger). Passer aux **hidden states
   de Qwen** si on veut un latent aligné avec la politique (option A specs/01).

## Wiring dans la boucle (étape suivante, pas encore faite)

Le point de bascule est prêt côté interface : en Phase 2, un `JepaWorldModel` remplacera
`agents/world_model.py::WorldModel` en implémentant le **même** contrat
(`predict`, `score_to_goal`, `rollout`) mais **en latent** :
- `predict` → `model.predict_next(E_state(obs), E_state(action))` ;
- `divergence` → `1 - cos(ŝ', E_state(obs_réelle))` (remplace le proxy Jaccard de `surprise.py`) ;
- `score_to_goal` → distance latente à `E_state(goal)`.

L'orchestrateur (`loop.py`) ne change pas — seule l'implémentation du world-model devient latente.

## Limites assumées (v0)

- **Encodeur gelé** : ce n'est pas encore un H-JEPA (encodeur appris conjointement). C'est un
  *latent dynamics model sur embeddings gelés* — pragmatique, collapse-free, suffisant pour
  tester la thèse. Le passage à l'encodeur appris est un chantier ultérieur.
- L'apport réel se mesure **dans la boucle** (courbe réussite-vs-tours), pas à la perte de
  prédiction seule : une bonne perte ne garantit pas un meilleur agent.
