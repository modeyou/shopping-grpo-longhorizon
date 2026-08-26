# BPO runtime diagnostics

The BPO runtime keeps the training algorithm unchanged while recording three
diagnostic events in SHOPPING_GRPO_DIAGNOSTICS_PATH:

* bpo_actor_batch records response-mask token counts, non-zero policy-weight
  support, advantage/return support, and the tree audits for the accepted
  batch.
* bpo_actor_loss_batch records the loss_mask and attention_mask support seen
  by FSDPEngine.forward_backward_batch, together with the observed loss
  scalar. These masks are captured at the boundary before remove-padding
  packing. A zero loss_mask here means the actor cannot receive a gradient;
  a non-zero mask with a zero gradient points to the packing/Liger path.
* bpo_optimizer_backward records the existing gradient and parameter-delta
  audit for the optimizer update.

The loss audit is bounded by SHOPPING_BPO_LOSS_AUDIT_LIMIT (default 8) so
formal runs do not grow the diagnostics file without bound. Diagnostic writes
are best-effort and never change the loss or optimizer path.

For a one-step validation run, set:

    export SHOPPING_BPO_LOSS_AUDIT_LIMIT=8

Launch it through the explicit diagnostic contract. Do not override the formal
trainer step count directly:

    bash scripts/bpo.sh \
      --model "$BPO_MODEL" \
      --output "$BPO_DIAGNOSTIC_OUT" \
      --experiment-name "$BPO_DIAGNOSTIC_NAME" \
      --logger console \
      --shopper-model "$SHOPPER_MODEL" \
      --shopper-base-url "$SHOPPER_BASE_URL" \
      --seed 20260823 \
      --diagnostic-steps 1

The diagnostic still requires two accepted trees, non-zero gradients and an
actual parameter delta. It disables checkpointing and validation and must not
be promoted as a formal training result.

Then inspect training_diagnostics.jsonl for the three events in order:

    grep -E 'bpo_actor_batch|bpo_actor_loss_batch|bpo_optimizer_backward' \
      $BPO_SMOKE_OUT/training_diagnostics.jsonl
