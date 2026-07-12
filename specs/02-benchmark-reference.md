# 02 — Benchmark de référence

> Environnement d'évaluation de morpheus. **Décision figée : τ²-bench** (Sierra) comme bench réel de Phase 0.
> Données de leaderboard collectées **juillet 2026** — à re-vérifier avant publication (les scores bougent vite).

## Pourquoi τ²-bench

morpheus mesure une seule chose au fond : **la réussite de tâche en fonction du nombre de tours d'outils**, et cherche à redresser la courbe **au-delà de 8+ tours**. τ²-bench est le benchmark qui isole exactement ce régime :

- **tool-agent-user** : l'agent enchaîne des appels d'outils sur **plusieurs tours**, avec un **utilisateur simulé par LLM** et des **policy documents** à respecter ;
- **réel et reproductible** : outils/APIs de domaine réels, tâches d'entreprise vérifiables de bout en bout (pass@k) ;
- **domaines** : retail, airline, telecom — étendus en 2026 à des domaines **voice** et **knowledge-retrieval** ;
- c'est le successeur de τ-bench, la référence publiée la plus solide sur le tool-use multi-tours.

> Complément possible plus tard : **Terminal-Bench 2.1** (vraies tâches CLI agentiques) pour confirmer la thèse sur du coding réel. Non retenu en Phase 0 pour garder la courbe vs-tours lisible.

## Note de vocabulaire — noms de modèles réels (mi-2026)

Il n'existe **pas** de « Sonnet 4.7 / 4.8 » public. La lignée Anthropic mi-2026 :

- **Sonnet** : 4.6 → **Sonnet 5**
- **Opus** : 4.6 → 4.7 → **4.8**
- **Classe Mythos** (au-dessus d'Opus) : **Fable 5** (sortie 9 juin 2026, restaurée le 1er juil. 2026), **Mythos 5**

Toujours référencer ces noms dans les specs, pas des versions inexistantes.

## Cibles de référence — τ²-bench

Scores Claude publiés (retail / telecom), à battre / approcher :

| Modèle | Retail | Telecom | Remarque |
|---|---|---|---|
| **Mythos 5** | ~89,2 % (snapshot public) | — | classe Mythos, plafond actuel |
| **Fable 5** | ~89,2 % | — | classe Mythos |
| **Opus 4.6** | 91,9 % | 99,3 % | plafond Opus mesuré sur ce leaderboard |
| **Opus 4.5** | 88,9 % | 98,2 % | |
| **Sonnet 4.6** | 87,5 % (agrégé) | — | **cible principale morpheus single-GPU** |
| **Sonnet 4.5** | 0,862 (retail) / 0,700 (airline) | — | repère historique multi-tours |
| Sonnet 3.7 | — | ~49 % (pass@1) | ancienne génération |

> Les cases vides = non trouvées à la collecte ; à compléter depuis les leaderboards live avant de figer les cibles.

## La cible réaliste de morpheus

- **Barre haute inatteignable en single-GPU à court terme** : Opus 4.8 / Fable 5 / Mythos 5.
- **Cible utile et crédible** : **égaler ou approcher Sonnet 4.6** en réussite multi-tours, avec un **Qwen ~32B + orchestrateur JEPA** tenant sur ~32 Go VRAM.
- **Le vrai signal de succès n'est pas le score agrégé** mais **la forme de la courbe** : morpheus doit diverger favorablement de la baseline Qwen nue **précisément à 8+ tours**.

## Baselines open-weight (contexte)

Sur le tool-use multi-tours, les meilleurs open-weight sont des gros MoE (200B–1T) qui ne tiennent **pas** sur une carte : GLM-5, MiniMax M2.5, Kimi K2.5, DeepSeek V4. Le pari morpheus est de compenser l'écart de taille par l'**architecture orchestrateur** (boucle fermée + JEPA + RAG gated), pas par le scaling.

## Protocole de mesure (rappel, cf. doc 01)

Tracer **réussite de tâche vs longueur (tours : 4 → 8 → 12)** sur τ²-bench retail (puis telecom) pour chaque variante :
`baseline Qwen nue` · `+ boucle fermée (Ph.1)` · `+ JEPA MPC (Ph.2)` · `+ RAG gated (Ph.3)` · `+ routeur de surprise (Ph.4)`,
avec l'**API Sonnet 4.6** comme ligne de référence supérieure sur le même sous-ensemble.

## Sources (collecte juillet 2026 — à re-vérifier)

- [Agentic AI Benchmarks Leaderboard — GAIA, WebArena, BFCL, Tau2-Bench (Awesome Agents)](https://awesomeagents.ai/leaderboards/agentic-ai-benchmarks-leaderboard/)
- [Tau2 Telecom Leaderboard (llm-stats)](https://llm-stats.com/benchmarks/tau2-telecom)
- [TAU-bench 2026 tracked scores (BenchLM)](https://benchlm.ai/benchmarks/tauBench)
- [Terminal-Bench 2.1 Leaderboard 2026 (CodingFleet)](https://codingfleet.com/blog/terminal-bench-leaderboard-2026/)
- [SWE-bench Verified Leaderboard — July 2026 (BenchLM)](https://benchlm.ai/benchmarks/sweVerified)
- [SWE-bench Pro Leaderboard 2026 (Morph)](https://www.morphllm.com/swe-bench-pro)
- [Claude Fable 5 & Mythos 5 Benchmark Breakdown (Vellum)](https://www.vellum.ai/blog/claude-fable-5-and-mythos-5-benchmarks-explained)
- [AI Agent Leaderboard 2026 — 5 Benchmarks Ranked (Rapid Claw)](https://rapidclaw.dev/blog/ai-agent-benchmarks-2026)
