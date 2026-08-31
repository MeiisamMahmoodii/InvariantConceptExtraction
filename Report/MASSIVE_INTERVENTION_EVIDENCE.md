# MASSIVE intervention evidence

Frozen Gemma, partition, and joint decoder; no steering parameters were trained.

| Test | `zC` intervention | `zS` intervention |
|---|---:|---:|
| Donor factor follows swap | intent 61.6% | locale 97.8% |
| Other factor follows donor | locale 2.0% | intent 2.6% |
| Unrelated-label control | 0.67% | 2.18% |
| Generation steering | weak: donor intent 10% | partial: target language 19.5% at `alpha=0.75` |

## Interpretation

The partition is strongly actionable in representation space: swapping `zC` transfers donor intent while retaining locale, and swapping `zS` transfers donor locale while retaining intent. Surface steering partially survives into greedy generation; concept steering does not yet. Generation steering was stopped without tuning the layer, decoder, or alpha further.

## Evidence artifacts

- Representation C swap: `massive_joint_cs_decoder_c_swaps.json`
- Representation swap specificity and unrelated-label controls: `massive_joint_cs_decoder_swap_specificity.json`
- Joint reconstruction and S swaps: `massive_joint_cs_decoder_swaps.json`
- S-generation pilot and alpha sweep: `massive_s_steering_generation_pilot.json`, `massive_s_steering_alpha_sweep.json`
- Scaled S-generation evaluation (Wilson 95% CIs): `massive_s_steering_scale200.json`
- C-generation pilot: `massive_c_steering_generation_pilot.json`
