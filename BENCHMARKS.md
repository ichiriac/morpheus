# Résultats de bench morpheus

Journal cumulatif (une ligne par run). La métrique qui tranche = réussite **vs nombre de tours** ; la thèse veut voir la courbe *world-model* diverger de la baseline à 8+ tours.

> ### ⚠️ Les six lignes `tau2/retail` du 2026-07-13 sont ININTERPRÉTABLES — ne pas s'en servir
>
> Établi le 2026-07-15. Leur `0.0%` ne mesure ni Qwen, ni le dialogue, ni la boucle : il mesure un
> **404**. Le juge des NL-assertions n'a été câblé sur le vLLM local qu'à **19:43** (commit
> `1158aa0`) ; avant, le défaut τ² était `gpt-4.1`, sans clé ⇒ 404 ⇒ composante NL = 0. Or
> **112/114 tâches retail** ont une `NL_ASSERTION` dans leur `reward_basis` ⇒ `reward = db × 0 = 0`
> **par construction**, quoi que l'agent ait fait. Les six runs sont tous ANTÉRIEURS à 19:43.
>
> Recoupement : le **seul** signal non nul du 13/07 est `tau2/telecom` en mode **solo** (33% / 50% /
> 100%) — précisément le seul mode dont le reward NE passe PAS par le juge NL.
>
> Ce n'est PAS la « limite d'équité » qu'invoquaient `qwen_tau2.yaml` et `qwen_tau2_jepawm.yaml` :
> cet avertissement-là était lui-même périmé (`respond_to_user` livré le 13/07 à 12:26, commit
> `7d6349d`, soit AVANT tous ces runs ; 44 des 108 tours annotés (41%) sont des `respond_to_user`).
> Les lignes sont conservées telles quelles — on n'efface pas un journal — mais elles ne disent rien
> sur la performance. **On ne sait toujours pas ce que vaut Qwen nu sur retail.**

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
