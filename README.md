# morpheus

**M**odèle à **O**ntologie **R**éelle par **P**erception **H**iérarchique d'un **E**space **U**niversel de **S**ens.

![Morpheus tend les deux pilules à Neo](specs/morpheus-the-choice.webp)

> *« This is your last chance. After this, there is no turning back. »*

Deux pilules. Deux façons de croire qu'une machine peut *juger*.

**La pilule bleue** — le confort. On admet que l'agentique longue tâche restera l'apanage des modèles frontier fermés et des MoE de 200B à 1000 milliards de paramètres, tournant sur des fermes de GPU. On appelle une API, on ne cherche pas à comprendre, et le monde reste tel qu'on nous le montre.

**La pilule rouge** — le pari de morpheus. On regarde jusqu'où va le terrier du lapin : tenir des tâches longues (**10+ tours d'outils**) sur **un seul GPU (< 5000 €, ~32 Go VRAM)**, non pas en gonflant le modèle, mais en lui donnant un *monde intérieur* pour planifier.

## Le pari

Le goulot pour égaler Sonnet 4.5 / Opus 4.8 en agentique n'est **plus la syntaxe d'appel d'outil** — un ~32B la maîtrise déjà. Ce qui manque, c'est la **couche de jugement multi-tours** : savoir quand appeler un outil (et quand ne pas le faire), garder le fil de l'objectif sur 10-15 appels, récupérer intelligemment après une erreur.

morpheus garde **Qwen comme politique** et lui adjoint un **world-model latent de type JEPA** qui simule les conséquences d'une action *avant* de l'exécuter, en **boucle fermée** (MPC à horizon glissant) : imaginer un plan, exécuter un pas, ré-ancrer sur l'état *réel*, replanifier. L'erreur ne se compose pas — chaque tour ré-ancre sur la vérité.

## Le nœud à craquer

Un pic d'erreur de prédiction — la **surprise** — ne dit pas *pourquoi* il survient :

- **« j'ai fauté »** → une erreur à corriger ;
- **« le monde est plus riche que mon plan »** → c'est mon plan qui avait tort.

Les deux produisent la même surprise. Le cœur scientifique de morpheus est le **routeur de surprise** qui désambiguïse ces deux régimes et déclenche en conséquence la récupération de connaissance (**RAG gated par la divergence**) — le seul mécanisme capable d'attraper la classe d'erreurs *cohérentes-mais-fausses*, structurellement invisibles pour JEPA.

## Où regarder

| Document | Contenu |
|---|---|
| [specs/00-contexte-experience.md](specs/00-contexte-experience.md) | D'où vient l'idée, les faits qui la cadrent, la thèse, le vrai problème ouvert. |
| [specs/01-orchestrateur-jepa-qwen.md](specs/01-orchestrateur-jepa-qwen.md) | L'architecture : rôles Qwen / JEPA / RAG, boucle fermée, routeur de surprise, prototype minimal. |
| [specs/02-benchmark-reference.md](specs/02-benchmark-reference.md) | L'évaluation : τ²-bench (figé), cibles chiffrées, protocole de mesure. |

**Bench de départ** : τ²-bench (retail puis telecom). **Cible réaliste** : approcher Sonnet 4.6 sur un seul GPU — pas le décrocher, l'*approcher*, ce qui serait déjà une rupture.

*Remember: all I'm offering is the truth. Nothing more.*
