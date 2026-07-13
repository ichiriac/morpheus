# Contenu RAG — base de connaissance par domaine

Corpus interrogé par le RAG *gated par la surprise* (`agents/knowledge.KnowledgeBase`,
retriever BM25 ; loop.py étape 5 récupère quand δ dépasse le seuil). C'est le **référentiel de
vérité** qui attrape le *cohérent-mais-faux* que JEPA seul ne voit pas (cf. specs/00, specs/01).

## Fichiers

- `retail.md`, `airline.md` — un fichier par domaine, chunké en règles atomiques :
  - **Politique du domaine** (règles conditionnelles « on ne peut X que si Y ») ;
  - **Signatures des outils** (`nom(arg1*, arg2)` avec les VRAIS noms d'arguments, `*` = requis) —
    chaque outil = une règle distincte pour le retriever.

## Provenance & régénération

Dérivé de [tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT © 2025 Sierra Research).
**Généré** — ne pas éditer à la main :

```bash
python scripts/build_kb.py            # retail + airline
```

Le runner **préfère** `data/kb/<domaine>.md` (versionné, reproductible, sans dépendance à
l'install τ²) ; à défaut il retombe sur le `policy.md` brut de τ² (`agents/knowledge.locate_policy`).

## Notes

- **telecom** : pas encore inclus — domaine dual-control sans `policy.md` unique (connaissance en
  workflow τ²), à dériver séparément.
- Les signatures d'outils sont aussi injectées dans le `system_context` de la politique au pas
  PROPOSER (cf. `envs/tau2_adapter`) ; leur présence dans le RAG sert la récupération *sur surprise*
  (replanification), pas la proposition initiale.
