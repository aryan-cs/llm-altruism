# Paper Summary

## Experiment Summary

| experiment_id | track | presentation | part | game | trial_count | cooperation_rate_a | cooperation_rate_b | average_payoff_a | average_payoff_b | total_duration_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper-baseline-prisoners_dilemma-20260407T172819Z | baseline | canonical | 1 | prisoners_dilemma | 18 | 0.4815 | 0.5370 | 2.2037 | 1.9259 | 1109.5061 |
| paper-baseline-prisoners_dilemma-20260407T173331Z | baseline | canonical | 1 | prisoners_dilemma | 18 | 0.4722 | 0.5463 | 2.2500 | 1.8796 | 667.0035 |

## Prompt Variant Breakdown

| experiment_id | track | presentation | game | prompt_variant | trial_count | cooperation_rate_a (95% CI) | cooperation_rate_b (95% CI) | average_payoff_a (95% CI) | average_payoff_b (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper-baseline-prisoners_dilemma-20260407T172819Z | baseline | canonical | prisoners_dilemma | minimal-neutral | 6 | 0.2778 [0.0833, 0.5833] | 0.2778 [0.0556, 0.6111] | 1.6111 [1.0556, 2.3056] | 1.6111 [1.1667, 2.2500] |
| paper-baseline-prisoners_dilemma-20260407T173331Z | baseline | canonical | prisoners_dilemma | minimal-neutral | 6 | 0.2500 [0.0556, 0.5556] | 0.3056 [0.0833, 0.6118] | 1.7500 [1.2222, 2.3333] | 1.4722 [1.0278, 2.1389] |
| paper-baseline-prisoners_dilemma-20260407T172819Z | baseline | canonical | prisoners_dilemma | minimal-neutral-compact | 6 | 0.5833 [0.3056, 0.8611] | 0.6667 [0.4444, 0.8889] | 2.5000 [2.1667, 2.8333] | 2.0833 [1.4722, 2.6944] |
| paper-baseline-prisoners_dilemma-20260407T173331Z | baseline | canonical | prisoners_dilemma | minimal-neutral-compact | 6 | 0.5833 [0.3056, 0.8611] | 0.6667 [0.4444, 0.8889] | 2.5000 [2.1667, 2.8333] | 2.0833 [1.4722, 2.6944] |
| paper-baseline-prisoners_dilemma-20260407T172819Z | baseline | canonical | prisoners_dilemma | minimal-neutral-institutional | 6 | 0.5833 [0.3056, 0.8611] | 0.6667 [0.4444, 0.8889] | 2.5000 [2.1667, 2.8333] | 2.0833 [1.4722, 2.6944] |
| paper-baseline-prisoners_dilemma-20260407T173331Z | baseline | canonical | prisoners_dilemma | minimal-neutral-institutional | 6 | 0.5833 [0.3056, 0.8611] | 0.6667 [0.4444, 0.8889] | 2.5000 [2.1667, 2.8333] | 2.0833 [1.4722, 2.6944] |

## Pooled Prompt Variant Summary

| track | presentation | game | prompt_variant | experiment_count | trial_count | cooperation_rate_a (95% CI) | cooperation_rate_b (95% CI) | average_payoff_a (95% CI) | average_payoff_b (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | canonical | prisoners_dilemma | minimal-neutral | 2 | 12 | 0.2639 [0.0972, 0.4722] | 0.2917 [0.1111, 0.5000] | 1.6806 [1.2635, 2.1389] | 1.5417 [1.1806, 1.9444] |
| baseline | canonical | prisoners_dilemma | minimal-neutral-compact | 2 | 12 | 0.5833 [0.3750, 0.7917] | 0.6667 [0.5000, 0.8333] | 2.5000 [2.2500, 2.7500] | 2.0833 [1.6250, 2.5417] |
| baseline | canonical | prisoners_dilemma | minimal-neutral-institutional | 2 | 12 | 0.5833 [0.3750, 0.7917] | 0.6667 [0.5000, 0.8333] | 2.5000 [2.2500, 2.7500] | 2.0833 [1.6250, 2.5417] |
