# Contenu RAG — base de connaissance par domaine

Corpus interrogé par le RAG *gated par la surprise* (`agents/knowledge.KnowledgeBase`,
retriever BM25 ; loop.py étape 5 récupère quand δ dépasse le seuil). C'est le **référentiel de
vérité** qui attrape le *cohérent-mais-faux* que JEPA seul ne voit pas (cf. specs/00, specs/01).

## Fichiers

- `retail.md`, `airline.md`, `telecom.md` — un fichier par domaine, chunké en règles atomiques :
  - **Politique du domaine** (règles conditionnelles « on ne peut X que si Y ») ;
  - **Signatures des outils** (`nom(arg1*, arg2)` avec les VRAIS noms d'arguments, `*` = requis) —
    chaque outil = une règle distincte pour le retriever.
  - **telecom** ajoute le **manuel de dépannage** (`tech_support_manual.md` : « si pas de service →
    airplane mode / SIM / APN / suspension… ») — la connaissance procédurale à récupérer sur surprise.

## Provenance & régénération

Dérivé de [tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT © 2025 Sierra Research).
**Généré** — ne pas éditer à la main :

```bash
python scripts/build_kb.py            # retail + airline
```

Le runner **préfère** `data/kb/<domaine>.md` (versionné, reproductible, sans dépendance à
l'install τ²) ; à défaut il retombe sur le `policy.md` brut de τ² (`agents/knowledge.locate_policy`).

## Notes

- **telecom** : domaine dual-control (pas de `policy.md` unique). Le KB combine `main_policy_solo.md`
  + `tech_support_manual.md`, et ses signatures (agent **+ device** : `toggle_airplane_mode`,
  `run_speed_test`…) viennent d'un reset gym SOLO (le constructeur d'env n'expose que les outils agent).
- Les signatures d'outils sont aussi injectées dans le `system_context` de la politique au pas
  PROPOSER (cf. `envs/tau2_adapter`) ; leur présence dans le RAG sert la récupération *sur surprise*
  (replanification), pas la proposition initiale.

## ⚠️ Garde-fous — non-contamination du benchmark τ²

Le RAG doit rester **comparable au leaderboard τ²** : il ne peut donner à l'agent que ce que le
vrai agent τ² a LÉGITIMEMENT. Vérifié : le `system_context` que τ² donne à l'agent telecom fait
**22 819 caractères et inclut déjà le manuel de dépannage** ; mon KB en est un **sous-ensemble**
(manuel inclus, **DB exclue**). Donc pas de fuite.

**Ce que le contenu RAG ne doit JAMAIS contenir** (sinon triche / score gonflé, non comparable) :

1. **La DB du domaine (`db.toml`)** — les VRAIS clients / commandes / lignes = les *réponses*.
   L'y mettre laisserait l'agent récupérer la réponse au lieu d'appeler l'outil. **Interdit absolu.**
2. **`user_scenario`** — le besoin CACHÉ de l'utilisateur (persona, `known/unknown_info`,
   `task_instructions`). Déjà exclu du `goal` (cf. `tau2_adapter._build_goal`).
3. **`evaluation_criteria`** — les actions de référence attendues.

**Confounding à connaître (pas une fuite, mais fausse la mesure)** : la policy est **déjà** dans le
`system_context` (elle y tient). Un RAG *sur la policy seule* est donc **redondant** — il ne fait
que re-surfacer une connaissance déjà présente ; une comparaison RAG vs sans-RAG n'isolerait pas
proprement sa valeur. Deux conséquences :

- **Ne PAS** retirer la policy du `system_context` du seul bras baseline pour « faire gagner » le RAG :
  ce serait handicaper le baseline → mesure trompeuse.
- La **mémoire épisodique** (`agents/memory.py`, `orchestrator.use_memory`) sort de ce régime : elle
  récupère des **faits atomiques issus des observations réelles de l'agent** — une connaissance qui
  N'EST PAS dans le `system_context` et qui excède la fenêtre de récence du transcript. C'est la
  voie fidèle à la thèse (specs/01 §RAG « faits atomiques en ligne ») et non contaminante (mémoire
  de ce que l'agent a lui-même observé, pas fuite).
