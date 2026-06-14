# Credits

Keryx-miner builds on the work of the wider Kaspa / kHeavyHash GPU-mining
community. Thanks in particular to:

- **tmrlvi** — original `v_dot8` implementation (upstream Kaspa).
- **BaikalMine** — stratum DAA score support (PR #6, commit `0f3a306`).
- **Suprnova fork (ocminer)** — Keccak round-loop unroll, sm_80 launch bounds,
  native sm_120 PTX dispatch, AMD `v_dot8` default + gfx906 JIT fix, and the
  per-arch auto workload sizing (ported in commit `191593e`).
- **VaniaHilkovets** — register-resident `eor3`/`bcax` `lop3` emission for the
  Keccak theta/chi steps (PR #7).

Upstream lineage: the CUDA/OpenCL kHeavyHash kernels derive from the original
Kaspa miner and its contributors.
