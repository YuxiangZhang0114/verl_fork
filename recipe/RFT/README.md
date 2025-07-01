# RFT: Reward Fusion with Token-Level Guidance

This recipe demonstrates how to implement the gated token reward mechanism
outlined in the repository discussion. The approach applies teacher guidance only
when the verifier score of a generated sequence is low **and** the policy
entropy has collapsed.

The token reward for each action token is computed as

```
r_t = log(p_teacher / p_ref) \
      * sigmoid(k_r * (tau_r - score)) \
      * sigmoid(k_H * (tau_H - entropy))
```

where `score` is the sequence level verifier score and `entropy` is the token
entropy of the current policy. The gating parameters `k_r`, `k_H`, `tau_r` and
`tau_H` are configured in `config/rft_trainer.yaml`.

Advantages are estimated using this token reward and combined with sequence
level rewards in the same way as the PRIME recipe. See `rft_core_algos.py` for
the implementation.

To launch a minimal run:

```bash
bash recipe/RFT/run_rft.sh
```

This code is largely based on the files under `recipe/prime` and reuses most of
the infrastructure provided there.
