# Résultats de bench morpheus

Journal cumulatif (une ligne par run). La métrique qui tranche = réussite **vs nombre de tours** ; la thèse veut voir la courbe *world-model* diverger de la baseline à 8+ tours.

> ### ⚠️ Les six lignes `tau2/retail` du 2026-07-13 : PARTIELLEMENT cassées, pas toutes
>
> **CORRIGÉ le 2026-07-15 (2e passe).** Une 1re version de ce bloc affirmait que ces zéros
> « mesuraient un 404 » et que `reward = db × 0 = 0` **par construction**. C'est FAUX pour la
> majorité des tâches, et l'erreur venait de faire porter au chiffre `112/114` une charge qu'il ne
> supporte pas. Mesuré depuis, dans `tau2.evaluator.evaluator_nl_assertions` :
>
> - `112/114` = tâches ayant `NL_ASSERTION` dans **`reward_basis`**. Ce n'est PAS ce qui déclenche
>   le juge.
> - Ce qui le déclenche, c'est **`nl_assertions` non vide** : seulement **40/114**. Les **74 autres**
>   prennent une sortie anticipée (`if not nl_assertions: return reward=1.0`) — `NL=1.0` rendu
>   **sans aucun appel LLM**, donc **sans 404 possible**, hier comme aujourd'hui. Vérité vacue :
>   rien à violer ⇒ tout est satisfait.
>
> Conséquence sur les 8 tâches des deux baselines : **5/8 sont vacues** (0, 1, 5, 6, 7) — leur
> reward valait `db × 1 = db`, donc leur `0%` signifiait **déjà `DB=0`, un vrai échec de Qwen,
> correctement mesuré**. Seules les tâches **2, 3, 4** appelaient le juge et ont pris le 404.
> Ces runs sont donc **hétérogènes** (5 mesures valides + 3 cassées), pas universellement nuls —
> c'est en tant qu'AGRÉGAT qu'ils restent ininterprétables.
>
> Ce qui tient de la 1re version : le juge n'a bien été câblé qu'à **19:43** (`1158aa0`), APRÈS tous
> ces runs ; et la « limite d'équité » invoquée par `qwen_tau2.yaml` / `qwen_tau2_jepawm.yaml` était
> bien un fantôme (`respond_to_user` livré à 12:26, `7d6349d`, AVANT ces runs ; 44 des 108 tours
> annotés (41%) en sont). Les lignes sont conservées — on n'efface pas un journal.
>
> Leçon : « mesure, ne crois pas les commentaires » vaut aussi pour ses propres conclusions. Le
> `112/114` a été recopié de la doc sans vérifier ce qu'il comptait.

| Date (UTC) | Run | Env / domaine | Mode | Variante | Modèle | K/H/Tmax | Tâches | Réussite | Courbe (tours:réussite) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-13 12:43 | `qwen_tau2_polctx_check` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 1 | ⚠️ 0.0% (pré-juge) | 5:0%(n1) |
| 2026-07-13 13:36 | `retail_sigcheck` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 1 | ⚠️ 0.0% (pré-juge) | 5:0%(n1) |
| 2026-07-13 13:37 | `telecom_fixed` | tau2/telecom | solo | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/6 | 3 | 33.3% | 0:33%(n3) |
| 2026-07-13 13:40 | `qwen_tau2_retail_fixed` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 6 | ⚠️ 0.0% (pré-juge) | 5:0%(n3) · 11:0%(n1) · 12:0%(n1) · 13:0%(n1) |
| 2026-07-13 14:30 | `retail_memcheck` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 1 | ⚠️ 0.0% (pré-juge) | 5:0%(n1) |
| 2026-07-13 14:33 | `retail_baseline_quick` | tau2/retail | user-sim | baseline | `Qwen/Qwen3-32B-AWQ` | 2/3/16 | 8 | ⚠️ 0.0% (pré-juge) | 5:0%(n3) · 6:0%(n2) · 11:0%(n1) · 12:0%(n1) · 13:0%(n1) |
| 2026-07-13 14:47 | `retail_baseline_quick2` | tau2/retail | user-sim | baseline | `Qwen/Qwen3-32B-AWQ` | 2/3/16 | 8 | ⚠️ 0.0% (pré-juge) | 5:0%(n3) · 6:0%(n2) · 11:0%(n1) · 12:0%(n1) · 13:0%(n1) |
| 2026-07-13 15:06 | `telecom_rag` | tau2/telecom | solo | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/6 | 2 | 50.0% | 0:50%(n2) |
| 2026-07-13 16:26 | `jepa_wm_smoke` | mock/retail | — | world-model | `stub` | 3/1/8 | 3 | 0.0% | 4:0%(n1) · 8:0%(n1) · 12:0%(n1) |
| 2026-07-13 17:38 | `qwen_tau2_telecom_solo_v2` | tau2/telecom | solo | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/6 | 3 | 100.0% | 0:100%(n3) |
| 2026-07-13 18:08 | `jepa_wm_tau2` | mock/retail | — | world-model | `stub` | 3/1/8 | 3 | 0.0% | 4:0%(n1) · 8:0%(n1) · 12:0%(n1) |
| 2026-07-15 20:26 | `retail_attrib` | tau2/retail | user-sim | baseline | `Qwen/Qwen3-32B-AWQ` | 4/1/16 | 3 | 0.0% — **DB=0 / NL=1.0**, ATTRIBUÉ ✅ | 5:0%(n2) · 11:0%(n1) |
| 2026-07-15 20:44 | `retail_cap2500` | tau2/retail | user-sim | baseline | `Qwen/Qwen3-32B-AWQ` | 4/1/16 | 3 | 0.0% | 5:0%(n2) · 11:0%(n1) |
<!-- BENCH:APPEND -->

> ### `retail_attrib` — le premier 0% ATTRIBUABLE (smoke d'attribution, 3 tâches)
>
> Contrairement aux 6 lignes du 13/07, ce zéro est décomposé. **Le juge n'est plus en cause** : il est
> câblé sur le vLLM local, il répond, il rend du parsable — `NL_ASSERTION = 1.0` sur les 3 tâches.
> **Le dialogue n'est pas en cause** : `respond_to_user` circule (2 / 2 / 7 émissions) et le user-sim
> répond. C'est la composante **DB qui est nulle** ⇒ `reward = 0 × 1 = 0`. Les 3 tâches finissent au
> plafond de 16 tours, aucune conclue.
>
> **Trois propriétaires, séparés — chacun son traitement :**
> 1. **Qwen (à MESURER, pas à réparer)** — il passe le NOM du produit là où un ID est attendu :
>    `get_product_details(product_id='Mechanical Keyboard')` → « Product not found », ×6. Au tour 11,
>    `[ÉTAT COURANT]` (jamais tronqué) contenait `"Mechanical Keyboard": "1656367028"` à l'offset 812
>    sur 1478 caractères : l'ID était **littéralement sous ses yeux**. Il émet aussi un placeholder
>    `<new_item_id_keyboard>` malgré l'interdit de `policy.py:21`. Motif sur les 3 tâches :
>    `get_product_details` ×6 · `get_order_details` ×10 · `respond_to_user` ×7 — la rubrique
>    `loop_no_progress`, en vrai. C'est le niveau réel du modèle : la baseline existe pour le montrer.
> 2. **Le harnais (à RÉPARER)** — après le tour 11, la table `nom → ID` n'est plus que dans le
>    scratchpad, tronqué à 600 caractères (`policy.py:30`) : la coupe tombe sur `"Headphones"…`,
>    l'ID à 812 a DISPARU. Le rattrapage devient structurellement impossible, même pour un agent
>    compétent. La fenêtre de TOURS est hors de cause (t10→t13 = 3 tours, cap à 8).
> 3. **Le juge — SONDÉ, et il ne discrimine pas.** Les 3 `NL_ASSERTION = 1.0` s'expliquent :
>    tâches 0 et 1 ont `nl_assertions` **vide** ⇒ 1.0 vacu, juge jamais appelé. La tâche 2 est le
>    SEUL appel réel de tout le smoke — assertion « *there are 10 t-shirt options available* »,
>    verdict **MET**, alors que l'agent a listé **9** options, n'a jamais prononcé « 10 », et a
>    conclu par « Some items are not available ». **Faux positif, 1 sur 1.**
>    ⇒ `NL_ASSERTION` est inexploitable en l'état : soit absent (74/114), soit complaisant.
>    **Lire le `DB` seul** — objectif, calculé par l'évaluateur, et sur 74/114 tâches c'est déjà
>    TOUT le reward. Corollaire : les 74 tâches vacues forment un banc PROPRE, sans juge du tout.
>
> Règle de la maison : aucun signal n'est cru tant qu'il n'a pas recalé un imposteur crédible.

> ### `retail_cap2500` — l'artefact d'éviction réparé, et le zéro qui RESTE
>
> Même config, mêmes 3 tâches, seul `_TRANSCRIPT_CHARS` change (600 → 2500, premier palier au-dessus
> du p95 mesuré = 2164 sur 1357 payloads τ²-retail ; le 600 ne couvrait que **38.6%** et tronquait
> **62.6%** des résultats — régime normal, pas cas limite).
>
> **L'artefact disparaît** : « NOM passé comme `product_id` » 6 → **0** · ID numérique correct 2 → 4 ·
> erreurs d'outil 14 → **5** · la tâche 1 cesse de boucler et conclut en **10** tours au lieu de 16.
> **Le `DB` reste 0.0 sur les 3 tâches.**
>
> ⇒ L'éviction à 600 caractères n'était PAS la cause du 0% : c'était du bruit par-dessus un vrai
> échec. **Qwen nu échoue ces 3 tâches retail pour de bon, artefact retiré.** C'est le premier
> chiffre de cette table qui dise quelque chose sur le modèle. n=3 : indicatif, pas une baseline.
>
> ⚠️ Le prompt de la politique a CHANGÉ (cap 600 → 2500) : les runs d'avant le 2026-07-15 ne sont
> pas comparables à ceux d'après.
>
> #### Taxonomie des 3 échecs DB — le cahier des charges de la fonction de coût
>
> Chaque tâche retail n'a **qu'UNE écriture DB** (tout le reste est de la lecture) : le `DB` se joue
> sur un seul appel. Diff entre l'écriture réelle et l'écriture de référence, run `retail_cap2500` :
>
> | Tâche | Famille | Écriture attendue vs réelle | `tool_error` |
> |---|---|---|---|
> | 0 | **C1 — ancrage FIN** | bonne commande, bons `item_ids`, bon paiement ; `new_item_ids` = `['2299424241','7747408585']` au lieu de `['7706410293','7747408585']` — **1 variante sur 2** | **False** |
> | 1 | **A — planification** | `exchange_delivered_order_items` **jamais tentée** (bloqué en collecte, 10 tours) | — |
> | 2 | **C2 — `coherent_but_wrong`** | attendu `return_delivered_order_items(#W2378156, [3 items])` ; réel = 2 écritures ABOUTIES sur `#W4776164` et `#W6679257` — bien formées, **mauvaises entités** | **False** |
>
> **La propriété commune, et c'est elle qui compte : AUCUNE de ces écritures n'erre.** Donc :
> - `tool_error` (le signal le plus fort du routeur) est **aveugle** aux trois ;
> - `divergence` est **aveugle** : le world-model prédit correctement le payload d'une action fausse
>   mais bien formée — il n'y a pas de surprise à détecter (cf. probe_predictor_form_vs_content, F5) ;
> - `score_to_goal(goal, state_text)` est **structurellement incapable** : après une écriture fausse
>   mais bien formée, le texte d'état EST une confirmation de succès plausible. Et le but est une
>   constante (1 valeur / 265 épisodes) ⇒ `cos(état, but)` ne peut pas encoder « quelle variante
>   l'utilisateur voulait ».
>
> ⇒ **Ce que `P(succès | état)` devra voir**, dérivé de ces 3 cas : (1) discriminer à la granularité
> d'UN argument (`7706410293` vs `2299424241` — deux variantes valides du même produit) ; (2)
> discriminer l'entité CIBLE (bon outil, mauvaise commande) ; (3) voir « l'écriture n'a pas eu lieu
> et les tours s'épuisent ». Les points 1 et 2 exigent de lire l'**ACTION**, pas seulement l'état.
> Ça recoupe la sonde C du 2026-07-15 : dispersion intra-état des scores = 0.055 — le MPC actuel
> n'a rien pour départager deux candidats qui ne diffèrent que par un identifiant de variante.
>
> #### Protocole de bench : les 74 tâches vacues, biais mesuré
>
> Sur les 74 tâches à `nl_assertions` vide, le reward officiel τ² **est** le `db` : aucun juge dans
> la boucle, aucune distorsion de protocole. Test de biais de sélection (74 vacues vs 40 jugées,
> AUC de Mann-Whitney) :
>
> | | 74 vacues | 40 jugées | AUC | |
> |---|---|---|---|---|
> | actions de référence | 4.72 | 5.03 | 0.470 | pas de biais |
> | écritures DB | 1.54 | 1.55 | 0.401 | pas de biais |
> | longueur consigne utilisateur | 533 | 643 | **0.339** | **biais réel** (z≈−2.8, p≈0.005) |
>
> ⇒ Les tâches jugées ont des consignes utilisateur **21% plus longues** — plus d'info détenue par
> l'utilisateur, donc plus de dialogue à extraire. Le banc des 74 **sous-échantillonne le dialogue** :
> biais CONNU et RAPPORTÉ, pas un bloqueur. Les 40 jugées restent une piste secondaire, flaggée
> « juge non validé », jusqu'à ce qu'un juge recale un imposteur.

> **Juge CONSTANT** : à partir du 2026-07-15, tous les bras d'une même comparaison (baseline / WM /
> Sonnet 4.6) doivent être scorés par le MÊME juge — le Qwen local. Qwen-juge-Qwen est faible dans
> l'absolu, mais constant entre les bras ⇒ les ÉCARTS restent interprétables. Ne jamais comparer un
> run jugé localement à un chiffre de leaderboard jugé `gpt-4.1`.

## Signal goal-relative — `score_to_goal` (Phase 2, fix du 2026-07-13)

Fix = terme d'**alignement but↔état** dans la perte JEPA (`goal_alignment_loss` : régression
`cos(proj(s),proj(g))≈2·progress−1` + InfoNCE état→but). Métrique = **étendue** de `score_to_goal`
(discrimination) et **monotonie** (H1) / **séparation succès-échec** (H2, `validate_goal_signal.py`).

| Checkpoint | Étendue synthétique | Monotonie (Spearman) | Étendue τ² (médiane intra-ép.) | Gate H1 | Gate H2 |
|---|---|---|---|---|---|
| `jepa_apigen` (AVANT, sans alignement) | **0.0086** (dégénéré) | ~0 | ~0.0086 | FAIL | **FAIL** (pentes ≈ 0.0001) |
| `jepa_apigen_goal` (APRÈS, MiniLM, APIGen) | **0.20** | **+0.96** | **0.176** | FAIL (rho −0.46) | **PASS** (p=0.0003, mais magnitude faible) |
| `jepa_tau2_align` v1 (held-out, 12 nég) | — | +0.567 | — | PASS (p=0.0001) | FAIL marginal (p=0.064) |
| **`jepa_tau2_align` v2 (held-out, 40 nég, prédicteur réel)** | — | **+0.634** | — | **PASS (p≈0)** | **PASS (p≈0)** |

> `jepa_tau2_align` : alignement entraîné sur transitions τ²-retail rejouées, **split held-out par
> trajectoire** (`build_tau2_alignment_data.py`, anti-leak). **v2** (validé, VERDICT ✅) : négatifs
> variés (`--neg-fracs 0.4,0.65,0.9` → 112 pos / 153 nég), **H2 length-robust** (NIVEAU moyen de
> `score_to_goal`, pas la pente OLS confondue par la longueur), prédicteur à VRAIES actions.
> Gate held-out (57 traj, 17 succ / 40 échecs) : **H1 PASS** rho +0.634 (p≈0) · **H2 PASS** niveau
> succès 0.637 > échec 0.545 (p≈0). JEPA-WM opérationnel en drop-in (`jepa_wm_tau2.yaml`) : divergence
> discrimine la vraie action (3/4 pas) ; classement rollout 1-pas bruité sous but générique (caveat).

**Lecture** : mécanisme validé **en-distribution** (étendue franche, monotone, discriminant). Sur
**τ²-retail**, H2 passe (séparation succès/échec, AVANT elle échouait) mais **H1 reste FAIL** (inversé) :
mismatch de distribution — but τ² = instruction générique en prose FR (pas un état-cible), states τ² =
blobs JSON bruts (pas des observations NL comme APIGen). Suite = entraîner l'alignement sur du domaine
τ² (held-out) OU but d'issue par tâche + states normalisés NL (cf. TODO « SUITE DU FIX »).
