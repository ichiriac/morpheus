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
