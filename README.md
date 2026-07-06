# Quadruped Locomotion with PPO (MuJoCo Ant)

Trained and evaluated quadruped policies with **Proximal Policy Optimization (PPO)** in **MuJoCo / Gymnasium**, comparing flat-ground baselines against **terrain adaptation**, **leg-amputation robustness**, and a **compound policy router**.

**Stack:** Python · PyTorch · Gymnasium · MuJoCo · Stable-Baselines3  
**Repo:** [github.com/FavouritePlayer/ant_sim](https://github.com/FavouritePlayer/ant_sim)

---

## Results at a glance

All primary benchmarks use **10 matched random seeds**, **1000-step episodes**, and **deterministic** policy actions unless noted.

### Terrain adaptation (`TerrainAnt-v0`, difficulty 0.4)

| Metric | Flat-trained (control) | Terrain-adapted (treatment) |
|---|---:|---:|
| Mean episode reward | 424 ± 106 | **893 ± 130** |
| Fall rate | 50% | **30%** |
| Mean episode length | 719 steps | **926 steps** |
| Mean forward velocity | 0.22 m/s | 0.10 m/s |

**Conclusion:** Training on randomized heightfields improves survival on unseen rough terrain. The terrain policy trades speed for stability.

### Leg damage — specialist policy (leg 1 / front-right amputated at reset)

| Metric | Flat-trained | Damage specialist |
|---|---:|---:|
| Mean episode reward | 44 ± 37 | **2148 ± 749** |
| Mean episode length | 21 steps | **809 steps** |
| Fall rate | 100% | **20%** |
| Mean forward velocity | −0.19 m/s | **0.31 m/s** |

**Conclusion:** A policy trained with leg 1 always amputated learns a stable tripod gait on that leg. It does **not** transfer to other legs (100% fall on legs 0, 2, 3).

### Leg damage — sudden amputation (leg 1 removed at step 120)

| Metric | Flat-trained | Damage specialist |
|---|---:|---:|
| Mean episode reward | 520 ± 21 | **1386 ± 758** |
| Fall rate | 100% | **80%** |
| Mean episode length | 133 steps | **504 steps** |

**Conclusion:** Mid-episode amputation is harder than fixed amputation. The specialist still beats the flat baseline but recovery is not guaranteed.

### Cross-leg training (random single-leg amputation during training)

| Leg removed at test | Cross-leg reward | Cross-leg fall | Specialist reward | Specialist fall |
|---:|---:|---:|---:|---:|
| 0 (back right) | **1491 ± 868** | **60%** | 50 ± 14 | 100% |
| 1 (front right) | 94 ± 74 | 100% | **1449 ± 924** | **50%** |
| 2 (front left) | 305 ± 255 | 100% | 54 ± 15 | 100% |
| 3 (back left) | 99 ± 28 | 100% | 70 ± 38 | 100% |

**Conclusion:** No single learned policy solves all legs. Cross-leg training helps leg 0; the leg-1 specialist remains best on leg 1.

### Compound router (pick specialist or cross-leg by amputated leg)

Macro eval over **4 legs × 10 seeds = 40 episodes** (fixed amputation at reset):

| Approach | Mean reward | Fall rate |
|---|---:|---:|
| Flat baseline | 89 ± 84 | 100% |
| Specialist only | 406 ± 760 | 88% |
| Cross-leg only | 497 ± 737 | 90% |
| **Compound router** | **836 ± 909** | **78%** |

Routing table: leg **0** → cross-leg, leg **1** → specialist, legs **2–3** → cross-leg.

**Conclusion:** Compound routing is the best overall damage strategy in this repo — an engineering switchboard, not a new unified policy. Legs 2 and 3 remain unsolved (100% fall for all approaches).

### Multi-seed training replication

| Profile | Seeds | Mean final eval reward |
|---|---:|---:|
| Terrain canonical | 0, 1, 2 | 428 ± 53 |
| Damage canonical | 0, 1, 2 | 2724 ± 409 |

Details: [`docs/assets/replications/SUMMARY.md`](docs/assets/replications/SUMMARY.md)

---

## Main conclusions (for resume / portfolio)

1. **Terrain RL generalizes:** terrain-trained policy achieves **~2.1×** the reward and **40% lower fall rate** vs flat baseline on unseen hills (difficulty 0.4).
2. **Damage robustness is achievable but leg-specific:** specialist policy reaches **~49×** flat reward with **20% fall** on the trained leg; **0%** transfer to other legs.
3. **Cross-leg training is partial:** fixes back-right (leg 0) but fails on the specialist's leg-1 niche.
4. **Compound routing beats any single policy:** **836 vs 497** mean macro reward and **78% vs 90%** fall vs cross-leg alone by switching checkpoints per amputated leg.
5. **Sudden amputation is an open problem:** even the best policies fall **80%** of the time when the leg is removed mid-episode.

---

## Resume bullet (copy-ready)

> Trained terrain-adapted and damage-robust MuJoCo quadruped policies with PPO. On unseen heightfield terrain: **893 ± 130 vs 424 ± 106** reward, **30% vs 50%** fall rate. Under front-right leg amputation: **2148 ± 749 vs 44 ± 37** reward, **20% vs 100%** fall rate, **0.31 m/s** tripod locomotion. Built multi-seed replication, extended eval sweeps, and a compound leg-routing controller (**836 ± 909** macro reward, **78%** fall vs **90%** for cross-leg-only).

---

## Demo videos

| Experiment | Video |
|---|---|
| Terrain: flat vs adapted | [comparison_demo.mp4](docs/assets/terrain/comparison_demo.mp4) |
| Damage: flat vs specialist (fixed leg 1) | [comparison_demo.mp4](docs/assets/damage/comparison_demo.mp4) |
| Damage: sudden leg 1 removal at step 120 | [sudden_amputation_demo.mp4](docs/assets/damage/sudden_amputation_demo.mp4) |
| Compound: cross-leg vs flat→specialist switch | [compound_comparison_demo.mp4](docs/assets/damage/compound_comparison_demo.mp4) |

Regenerate: `python generate_policy_videos.py`

---

## Repository layout

| Path | Purpose |
|---|---|
| `checkpoints/` | Committed flat, terrain, and damage specialist models |
| `envs/` | `TerrainAnt-v0` (heightfield) and `DamageAnt-v0` (amputation) |
| `compare_policies.py` | Terrain control vs treatment eval |
| `compare_damage.py` | Damage control vs treatment eval |
| `compound_damage.py` | Per-leg policy router + eval |
| `replicate_training.py` | Multi-seed canonical training chains |
| `docs/assets/` | Benchmark JSON, plots, and comparison videos |
| `tests/` | Regression tests for envs, metrics, and routing |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python compare_policies.py
python compare_damage.py
python compound_damage.py
python generate_policy_videos.py
python -m unittest discover -s tests -v
```

Committed checkpoints reproduce all headline numbers without retraining.

---

## Further reading

- [LEARNING.md](LEARNING.md) — RL concepts, environment design, and experiment methodology
- [checkpoints/README.md](checkpoints/README.md) — model provenance and training recipes

## License

MIT — see [LICENSE](LICENSE).
