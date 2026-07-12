# 01 — Orchestrateur JEPA + Qwen

> Comment intégrer, à côté de Qwen, un **modèle annexe de type JEPA** qui établit et réajuste le plan suite à une demande. C'est le cœur de morpheus.

## Principe directeur

**Aucune chirurgie sur Qwen.** Qwen reste un LLM autorégressif intact, utilisé comme **politique** (comprendre, proposer, verbaliser, appeler les outils). Le JEPA est un **module annexe, léger, entraîné séparément**, branché autour de Qwen comme un **planificateur / critique en espace latent**. On câble deux modèles complémentaires ; on n'en fusionne pas un.

Cela correspond au **Sens 2** du doc de contexte : *Qwen comme politique + world-model latent de l'environnement d'outils pour planifier*.

## Les quatre composants

```
                 ┌─────────────────────────────────────────────────┐
   demande ─────▶│  ORCHESTRATEUR morpheus (boucle fermée / MPC)    │
                 └─────────────────────────────────────────────────┘
                        │            ▲                 │
             propose    │            │ score / plan    │ exécute
             actions    ▼            │                 ▼
                 ┌────────────┐  ┌────────┐      ┌────────────┐
                 │   QWEN     │  │  JEPA  │      │   OUTILS    │
                 │ (politique)│  │ (world │      │ (fs, shell, │
                 │            │  │  model)│      │  http, MCP) │
                 └────────────┘  └────────┘      └────────────┘
                                     │  ▲
                       divergence ▲  │  │ état latent réel
                        (surprise) │  ▼
                                 ┌──────────────┐
                                 │  RAG / KB    │  ← gated par la surprise
                                 │ (vérité,     │
                                 │  faits)      │
                                 └──────────────┘
```

### 1. Qwen — la politique (`policy`)
- Parse la demande, propose **K actions candidates** au tour courant (tool calls avec paramètres), verbalise le raisonnement, exécute l'action retenue.
- Interface : `propose(state_text, goal, k) -> [action_1 … action_k]`.
- Ne planifie pas seul à long horizon : il **génère les candidats**, JEPA les **départage**.

### 2. JEPA — le world-model latent (`predictor` + `encoders`)
Trois sous-réseaux, recette JEPA classique :
- **Encodeur d'état** `s = E_state(observation)` — projette l'observation (texte d'état de l'env : arbo fichiers, réponse d'API, sortie de test…) dans l'espace latent « de ce qui compte ».
- **Encodeur d'action** `a = E_action(tool_call)` — encode l'appel d'outil envisagé.
- **Prédicteur** `ŝ' = P(s, a)` — prédit l'**état latent résultant** de l'action, **sans l'exécuter**.

Deux usages :
- **Lookahead / scoring** : pour chaque action candidate de Qwen, prédire `ŝ'` et mesurer la distance au **but latent** `g = E_state(goal)`. Retenir l'action qui rapproche le plus (planification MPC à horizon court `H`).
- **Signal de divergence** : après exécution réelle, comparer `ŝ'` (prédit) à `s'_réel = E_state(obs_réelle)`. `δ = dist(ŝ', s'_réel)` = **surprise**.

### 3. Outils — l'environnement
fs / shell / http / serveurs MCP. Chaque exécution renvoie une observation, ré-encodée par `E_state`. C'est **la réalité qui fait avancer d'un cran** (boucle fermée).

### 4. RAG / base de connaissance — le référentiel de vérité
- Récupération **gated par la surprise** : on n'interroge la KB **que quand `δ` dépasse un seuil**.
- Rattrape la classe d'erreurs que JEPA ne peut pas voir : le **cohérent-mais-faux**.
- Style LWM-Planner : extrait aussi en ligne des **faits atomiques** de l'expérience courante pour enrichir la KB.

## La boucle fermée (algorithme)

```
morpheus(demande):
    g   = E_state(demande)                      # but latent
    s   = E_state(observe_env())                # état latent courant
    plan = qwen.esquisse_plan(demande)          # A→B→C→…→F, révisable
    pour t dans 1..T_max:
        # 1. PROPOSER
        candidats = qwen.propose(s_text, plan, k=K)

        # 2. LOOKAHEAD (MPC horizon H, dans le latent, sans exécuter)
        pour c dans candidats:
            score[c] = rollout_latent(s, c, horizon=H, but=g)   # Σ dist prédite au but
        a* = argmin(score)

        # 3. EXÉCUTER un seul pas (réalité)
        obs = exécute(a*)
        s'_réel = E_state(obs)

        # 4. DIVERGENCE
        ŝ'   = P(s, E_action(a*))
        δ    = dist(ŝ', s'_réel)

        # 5. ROUTER LA SURPRISE  (le nœud à craquer)
        si δ > seuil:
            type = routeur_surprise(δ, s, a*, ŝ', s'_réel)
            si type == ERREUR:      plan = qwen.replanifie(s'_réel, plan, faits=rag(s'_réel))
            si type == NOUVEAUTÉ:   plan = qwen.assimile(s'_réel, plan, faits=rag(s'_réel))

        # 6. RÉ-ANCRER (boucle fermée : pas d'accumulation d'erreur)
        s = s'_réel
        si atteint(s, g): return succès
```

Points clés :
- **Étape 3** : on n'exécute **qu'un** pas, jamais le plan entier à l'aveugle → pas de dérive composée.
- **Étape 5** : c'est là que vit la contribution scientifique de morpheus (voir plus bas).
- **Étape 6** : `s ← s'_réel` (l'état vrai), jamais `s ← ŝ'` (le prédit).

## Le routeur de surprise (contribution centrale)

Le pic `δ` dit **qu'**il y a surprise, pas **de quelle nature**. morpheus doit trancher **ERREUR** (j'ai fauté, corriger) vs **NOUVEAUTÉ** (le monde est plus riche que mon plan, assimiler). Signaux candidats à combiner, **au-delà de l'amplitude `δ`** :

| Signal | Intuition | ERREUR si… | NOUVEAUTÉ si… |
|---|---|---|---|
| **Direction dans le latent** | l'écart éloigne-t-il ou non du but `g` ? | `s'_réel` s'éloigne de `g` | `s'_réel` reste aligné vers `g` |
| **Cohérence RAG** | l'état vrai contredit-il un fait connu ? | contradiction avec la KB | cohérent, simplement absent du plan |
| **Signature de l'outil** | l'outil a-t-il renvoyé une erreur explicite (exit code, exception, 4xx/5xx) ? | oui | non (succès mais résultat inattendu) |
| **Réductibilité** | Qwen sait-il expliquer l'écart sans se contredire ? | non | oui |
| **Localité vs globalité** | l'écart casse-t-il la structure apprise (`E_state` incohérent) ou l'enrichit-il ? | rupture | extension |

→ Le routeur peut être un **petit classifieur** entraîné sur trajectoires annotées, ou un **prompt de jugement Qwen** alimenté par ces features. C'est le composant à instrumenter et mesurer en priorité.

## Comment entraîner le JEPA

- **Données** : trajectoires d'agent (état, action, état résultant) collectées sur l'environnement cible — τ²-bench (retail/airline) pour démarrer, puis traces de coding agentique réel (repo + shell + tests).
- **Objectif** : prédiction de représentation à la JEPA — `min dist(P(E_state(o), E_action(a)), sg(E_state(o')))`, avec `sg` = stop-gradient / cible EMA côté état résultant (éviter le collapse : VICReg / cible momentum, à la I-JEPA).
- **Ne pas** entraîner à reconstruire l'observation brute : l'environnement d'outils est **déjà symbolique** (un système de fichiers, une réponse d'API sont déjà abstraits). On prédit **dans l'abstrait de ce qui compte pour le but**, pas le texte exact.
- **Taille** : commencer **petit** (~100M–1B). Le JEPA est un module annexe ; l'essentiel du raisonnement reste chez Qwen. Le scaling JEPA n'est pas prouvé au-delà de ~1,2B → ne pas en dépendre.
- **Le but latent `g`** : conditionner le prédicteur sur le but rend l'espace « de ce qui compte » **goal-relative** — c'est ce qui aide le routeur à juger « plus près / plus loin ».

## Roadmap — du prototype au système

### Phase 0 — Baseline nue (mesure de référence)
Qwen3 32B seul, boucle ReAct classique, sur τ²-bench retail. **Mesurer la réussite en fonction du nombre de tours (4 → 8 → 12).** C'est la courbe qu'on veut redresser.

### Phase 1 — LLM-as-world-model (aucun entraînement)
Avant toute chirurgie : **Qwen lui-même** joue le world-model (simule les états/récompenses en texte), lookahead récursif type LWM-Planner / RAP / MCTS. Vérifie que **la boucle fermée + lookahead** apporte déjà un gain. Sépare le mérite « boucle » du mérite « JEPA ».

### Phase 2 — JEPA latent + MPC court
Entraîner le petit JEPA sur les trajectoires de Phase 0/1. Remplacer le lookahead-texte par un **rollout latent** (horizon `H = 2–4`). Objectif : gain **et** coût inférence réduit vs Phase 1.

### Phase 3 — Divergence + RAG gated
Brancher `δ` et le RAG déclenché par la surprise. KB = faits du domaine + faits atomiques extraits en ligne.

### Phase 4 — Routeur de surprise
Instrumenter les signaux du tableau ci-dessus, entraîner/évaluer le routeur ERREUR vs NOUVEAUTÉ. **C'est la mesure qui valide (ou non) la thèse morpheus.**

## Le protocole de mesure qui tranche

Sur τ²-bench (puis coding agentique), tracer la **réussite de tâche vs longueur (tours)** pour :
`baseline nue` · `+ boucle fermée (Ph.1)` · `+ JEPA MPC (Ph.2)` · `+ RAG gated (Ph.3)` · `+ routeur (Ph.4)`.

**Hypothèse à falsifier** : les courbes JEPA divergent favorablement de la baseline **précisément à partir de 8+ tours** (là où la baseline s'effondre). Si le gain n'apparaît qu'à horizon court et s'évapore à 12 tours, l'objection « myopie / évaluation OOD » l'emporte et il faut passer à un world-model **hiérarchique multi-échelle** (vision H-JEPA de LeCun) — un chantier à part.

## Décisions

1. **Environnement de départ** — **FIGÉ : τ²-bench** (Sierra), domaine retail d'abord puis telecom. Bench réel, multi-tours, isole exactement le régime 10+ tours. Détails et cibles chiffrées : [02-benchmark-reference.md](02-benchmark-reference.md). Ligne de référence supérieure sur le même sous-ensemble : **API Sonnet 4.6**.

2. **Stack de la boucle** — **FIGÉ : maison (Python), + DSPy plus tard.** La boucle de contrôle
   MPC reste écrite à la main ([orchestrator/loop.py](../src/morpheus/orchestrator/loop.py))
   pour les Phases 1→3 : elle EST la contribution (routeur de surprise, RAG gated), épouse le
   world-model latent (le rollout JEPA n'est pas un « tool call »), reste déterministe et
   déjà instrumentée (`TraceStep`/tour). **Pas** de LangGraph pour l'instant (paierait seulement
   avec un besoin de persistance/branchement/produit). **DSPy** est réservé à une **couche
   orthogonale** : optimiser automatiquement le prompt de la politique Qwen contre la métrique
   réussite-vs-tours (à introduire en Phase 1/3, après une baseline). PyTorch reste pour le seul JEPA.

3. **Runtime Qwen** — **FIGÉ : vLLM, GPU retenu RTX A6000 48 Go.** vLLM pour le débit-batching
   (les rafales d'appels du MPC), l'endpoint OpenAI déjà câblé, et le contrôle propre du thinking
   Qwen3. Modèle : **Qwen3-32B-AWQ** (4-bit) avec `MAX_LEN=32768` (les 48 Go donnent un gros cache
   KV et un batch large). A6000 = Ampere → **pas de FP8** ; option qualité = GPTQ 8-bit (~34 Go).
   Détail et table par VRAM : [04-runpod-qwen.md](04-runpod-qwen.md).

### Encore ouvertes
4. **Faut-il implémenter Phase 1 (LLM-as-WM) avant d'écrire une ligne de JEPA** — recommandé, pour isoler la valeur de la boucle fermée.
