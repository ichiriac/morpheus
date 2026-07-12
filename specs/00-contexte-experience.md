# 00 — Contexte de l'expérience

> Distillation de la conversation d'origine. Sert de socle partagé pour toutes les décisions d'architecture qui suivent.

## Le point de départ

Question initiale : **d'ici 3 à 9 mois, pourra-t-on faire tourner sur un seul GPU (< 5000 €) un modèle de coding aussi performant qu'Opus 4.8 ?**

La réflexion a dérivé du *coding pur* vers ce qui compte réellement en usage agentique : **l'utilisation d'outils sur des tâches longues (10+ tours)**. C'est là que se situe le vrai écart, et c'est l'objet de morpheus.

## Les faits qui cadrent le problème

### 1. Le plafond open-weight frôle le frontier fermé — mais avec des modèles énormes
Début 2026, les leaders open-weight sur SWE-bench Verified (MiniMax M2.5 ~75,8 %, GLM-5 ~72,8 %, Kimi K2.5 ~70,8 %, DeepSeek V4) approchent le niveau frontier. Mais ce sont des MoE de **200B à ~1T paramètres** qui tournent sur des serveurs GPU, pas sur une carte.

### 2. La vraie contrainte est la VRAM, pas le compute
« Un seul GPU < 5000 € » ≈ **~32 Go VRAM** (classe RTX 5090). En quantization Q4 (~0,6 Go/milliard), ça plafonne autour de **30B dense**. Ce qui rentre : Qwen3-Coder ~30B, Gemma 27B, Devstral, Qwen3 32B — d'excellents modèles, un cran sous le plafond open-weight, lui-même sous Opus.

### 3. Le piège du MoE
Les paramètres *actifs* d'un MoE déterminent la **vitesse**, mais il faut charger **tous** les poids en VRAM. Un 80B-A3B est rapide comme un 3B mais occupe la mémoire d'un 80B. Le MoE résout le compute, pas le « tenir sur une carte ».

### 4. Le tool-use se mesure à deux niveaux distincts
- **Niveau « appel » (BFCL)** : bonne fonction, bons types, JSON valide. → **Déjà égalé** sur un seul GPU (Qwen3 32B ~75,7 %, GLM-4.5 ~76,7 %). Barre franchie.
- **Niveau « tâche » (τ-bench / τ²-bench)** : simulation multi-tours réaliste, réussite de bout en bout. → **Sonnet 4.5 domine encore nettement** (~0,700 airline, ~0,862 retail). C'est ici que se joue tout.

## Le diagnostic central

Le goulot pour égaler Sonnet 4.5 en agentique **n'est plus la syntaxe d'appel** — c'est la **couche de jugement agentique** :

- savoir **quand** appeler un outil, et surtout quand **ne pas** le faire ;
- la **cohérence sur plusieurs tours** (garder le fil de l'objectif sur 10-15 appels) ;
- la **récupération après erreur** (retry intelligent au lieu de boucler) ;
- la **précision des paramètres** sur des chaînes multi-hop.

Cette couche corrèle aujourd'hui avec l'échelle **et surtout avec le post-training RL agentique** sur des trajectoires d'outils réelles. C'est de là que vient l'avantage de Sonnet 4.5 — pas d'une architecture magique.

## L'hypothèse morpheus : Qwen + JEPA

> Un Qwen3 32B augmenté d'un **world-model latent de type JEPA** pourrait battre un Qwen 32B nu sur les tâches longues, et débloquer l'agentique single-GPU.

### Deux sens à ne pas confondre
- **Sens 1 (à écarter)** — réentraîner le *backbone* de Qwen avec un objectif JEPA (prédire des représentations au lieu de tokens). Ne cible pas le point faible ; scaling JEPA non prouvé (plus gros JEPA publié ~1,2B).
- **Sens 2 (la bonne lecture)** — garder Qwen comme **politique** et lui **adjoindre** un world-model latent de l'environnement d'outils pour **planifier**. Simuler les conséquences d'une action *avant* de l'exécuter : « si j'appelle cet outil avec ces paramètres, voici l'état latent où j'atterris » — sans exécuter réellement.

### Pourquoi ça devrait aider sur 10+ tours
1. **Lookahead avant action** — dérouler mentalement plusieurs séquences dans le latent (type MPC) et choisir la meilleure, au lieu d'appeler « au feeling ».
2. **Signal de divergence pour la récupération** — un pic d'erreur de prédiction = détecteur gratuit de « ça ne s'est pas passé comme prévu ».

## Le raffinement décisif (apporté au fil de la discussion)

La première objection — *« la dérive de prédiction s'accumule sur 10 pas »* — est un **mauvais procès**. Elle ne vaut que pour la planification en **boucle ouverte** (imaginer 10 pas d'un coup puis exécuter à l'aveugle).

En **boucle fermée** (MPC à horizon glissant, = l'expérience empirique humaine) :
- on imagine un plan, on exécute **un** pas, on regarde où on est **réellement** arrivé, puis on **replanifie depuis la vraie position** ;
- c'est la **réalité** — pas la prédiction — qui fait avancer d'un cran ; l'imagination ne sert qu'à **choisir** la prochaine action ;
- donc **l'erreur ne se compose pas** : chaque tour ré-ancre sur l'état vrai. Le « je suis arrivé à C au lieu de B » **est** ce ré-ancrage.

## Le vrai problème ouvert (reformulé, corrigé)

Ce n'est **pas** la dérive de prédiction. Ce sont deux résidus :

1. **Myopie (optimum local glouton)** — pour choisir le bon prochain pas, il faut imaginer assez loin pour voir qu'un chemin mène à une impasse 4 pas plus tard. Si l'imagination n'est fiable qu'à 2 pas, la boucle fermée n'empêche pas d'entrer un pas à la fois dans un piège globalement condamné.

2. **Évaluation d'états surprenants / hors-distribution** — « je suis à C au lieu de B, qu'est-ce que ça change et comment réajuster ? » est **elle-même une prédiction**, et la plus dure : C est précisément un état inattendu, potentiellement OOD. Dissymétrie clé :
   - le **signal de divergence** (« il y a erreur ») est **facile et fiable** ;
   - **comprendre ce que l'erreur change** et en déduire le réajustement demande de refaire tourner le world-model sur un état surprenant — le régime où ces modèles s'effondrent empiriquement (cf. `stable-worldmodel` : chute brutale sous perturbation mineure).

## Le rôle de la connaissance (RAG), sous-estimé

- Un juge de cohérence entraîné en self-supervised juge par rapport à sa **distribution d'entraînement**, pas à la **vérité**. « Ça cohère » = « ça ressemble à la structure du monde apprise », pas « c'est correct ».
- **JEPA attrape l'incohérence statistique** (erreurs grossières, hallucinations qui cassent la structure) mais est **structurellement aveugle au cohérent-mais-faux** (état plausible, bien formé, factuellement erroné).
- **Le RAG bouche exactement ce trou** : seul mécanisme qui attrape la classe d'erreurs que JEPA ne peut pas voir. Ce n'est pas un confort — c'est le **référentiel de vérité** contre lequel le juge de cohérence vérifie.

### L'économie élégante : RAG *gated* par la surprise
On ne récupère pas en permanence (coûteux) : **on récupère quand le signal de divergence se déclenche**. La surprise devient le déclencheur du RAG. Architecture réelle et propre (cf. LWM-Planner qui extrait en ligne des « faits atomiques » de sa propre expérience ; « retrieval-augmented world models »).

### L'imbrication finale
Entraînement et récupération ne sont **pas** concurrents, ils sont **imbriqués** :
- le **RAG a besoin du juge entraîné** pour savoir *quoi* aller chercher ;
- le **juge a besoin du RAG** pour vérifier ce qu'il ne peut pas savoir.
- Point d'attache de la boucle : **le signal de divergence**.

## Le nœud à craquer

> **Désambiguïser la surprise.** Un pic d'erreur de prédiction ne distingue pas *« je viens de fauter »* (erreur à corriger) de *« je suis tombé sur quelque chose de nouveau mais légitime »* (c'était mon plan qui avait tort). Les deux produisent la même surprise.

La question qui vaut un papier — et qui définit morpheus :
**quel signal, en plus de l'amplitude de l'erreur, permet de séparer « j'ai fauté » de « le monde est plus riche que mon plan », et de router vers la bonne connaissance en conséquence ?**

## Verdict de faisabilité (rappel honnête)

- **Benchmarks structurés étroits** (HumanEval+, sous-ensemble SWE-bench) : oui, ~3-9 mois, scores comparables plausibles (gamiables par spécialisation).
- **Agentique réelle multi-tours** sur une seule carte : plutôt **12-24 mois**, autant par l'arrivée de matériel à mémoire unifiée plus large que par la compression seule.
- Les world-models sont un **pari complémentaire** au RL agentique, pas un remplaçant démontré. Le champion empirique actuel (Sonnet 4.5) tient son avance du RL sur trajectoires réelles, pas d'un world-model.

morpheus assume ce pari : **explorer l'axe world-model latent + RAG gated par la surprise comme voie vers l'agentique single-GPU.**
