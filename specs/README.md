# Specs — morpheus

**M**odèle à **O**ntologie **R**éelle par **P**erception **H**iérarchique d'un **E**space **U**niversel de **S**ens.

Ce dossier documente le contexte et l'architecture cible du projet : construire un **orchestrateur agentique** capable de tenir des tâches longues (10+ tours d'outils) sur un **seul GPU (< 5000 €, ~32 Go VRAM)**, en couplant un LLM open-weight (Qwen) à un **world-model latent de type JEPA** qui établit et réajuste le plan.

## Documents

| Fichier | Contenu |
|---|---|
| [00-contexte-experience.md](00-contexte-experience.md) | Le contexte de l'expérience : d'où vient l'idée, les faits qui la cadrent, la thèse centrale, le vrai problème ouvert. |
| [01-orchestrateur-jepa-qwen.md](01-orchestrateur-jepa-qwen.md) | L'architecture de l'orchestrateur : rôles Qwen / JEPA / RAG, la boucle fermée, le routeur de surprise, et un prototype minimal testable. |
| [02-benchmark-reference.md](02-benchmark-reference.md) | L'environnement d'évaluation : τ²-bench (figé), noms de modèles réels mi-2026, cibles chiffrées, protocole de mesure, sources. |
| [03-scaffold.md](03-scaffold.md) | L'état du code : arborescence, démarrage rapide, correspondance code↔specs, ce qui est réel vs stubbé, prochaines étapes. |
| [04-runpod-qwen.md](04-runpod-qwen.md) | Étape 1 : brancher un vrai Qwen sur RunPod (vLLM) et valider le format de la politique avec `check-llm`. |

## Thèse en une phrase

Le goulot pour égaler Sonnet 4.5 / Opus 4.8 en agentique n'est plus la **syntaxe d'appel d'outil** (déjà résolue par un ~32B), mais la **couche de jugement multi-tours** — et un world-model JEPA en **boucle fermée**, qui planifie dans un espace latent et déclenche la récupération de connaissance (RAG) sur **signal de divergence**, est le pari le plus crédible pour débloquer ça sur un seul GPU.
