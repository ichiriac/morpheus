# Résultats de bench morpheus

Journal cumulatif (une ligne par run). La métrique qui tranche = réussite **vs nombre de tours** ; la thèse veut voir la courbe *world-model* diverger de la baseline à 8+ tours.

| Date (UTC) | Run | Env / domaine | Mode | Variante | Modèle | K/H/Tmax | Tâches | Réussite | Courbe (tours:réussite) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-13 12:43 | `qwen_tau2_polctx_check` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 1 | 0.0% | 5:0%(n1) |
| 2026-07-13 13:36 | `retail_sigcheck` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 1 | 0.0% | 5:0%(n1) |
| 2026-07-13 13:37 | `telecom_fixed` | tau2/telecom | solo | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/6 | 3 | 33.3% | 0:33%(n3) |
| 2026-07-13 13:40 | `qwen_tau2_retail_fixed` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 6 | 0.0% | 5:0%(n3) · 11:0%(n1) · 12:0%(n1) · 13:0%(n1) |
| 2026-07-13 14:30 | `retail_memcheck` | tau2/retail | user-sim | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/16 | 1 | 0.0% | 5:0%(n1) |
| 2026-07-13 14:33 | `retail_baseline_quick` | tau2/retail | user-sim | baseline | `Qwen/Qwen3-32B-AWQ` | 2/3/16 | 8 | 0.0% | 5:0%(n3) · 6:0%(n2) · 11:0%(n1) · 12:0%(n1) · 13:0%(n1) |
| 2026-07-13 14:47 | `retail_baseline_quick2` | tau2/retail | user-sim | baseline | `Qwen/Qwen3-32B-AWQ` | 2/3/16 | 8 | 0.0% | 5:0%(n3) · 6:0%(n2) · 11:0%(n1) · 12:0%(n1) · 13:0%(n1) |
| 2026-07-13 15:06 | `telecom_rag` | tau2/telecom | solo | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/6 | 2 | 50.0% | 0:50%(n2) |
| 2026-07-13 16:26 | `jepa_wm_smoke` | mock/retail | — | world-model | `stub` | 3/1/8 | 3 | 0.0% | 4:0%(n1) · 8:0%(n1) · 12:0%(n1) |

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
| 2026-07-13 17:38 | `qwen_tau2_telecom_solo_v2` | tau2/telecom | solo | world-model | `Qwen/Qwen3-32B-AWQ` | 2/1/6 | 3 | 100.0% | 0:100%(n3) |
| 2026-07-13 18:08 | `jepa_wm_tau2` | mock/retail | — | world-model | `stub` | 3/1/8 | 3 | 0.0% | 4:0%(n1) · 8:0%(n1) · 12:0%(n1) |
