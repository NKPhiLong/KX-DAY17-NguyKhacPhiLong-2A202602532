Mode: offline (deterministic) | compact threshold: 600 tokens, keep 4 messages

## Standard Benchmark
conversations.json: 10 conversations, 101 turns, 14 recall questions

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 638 |                    14,203 |                     0% |               0.20 |                       0 |             0 |
| Advanced |                 588 |                    24,494 |                   100% |               1.00 |                     407 |             0 |

Advanced vs Baseline: 72% more prompt tokens processed, recall 0% -> 100%.

## Long-Context Stress Benchmark
advanced_long_context.json: 1 conversations, 16 turns, 3 recall questions

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 131 |                    21,748 |                     0% |               0.20 |                       0 |             0 |
| Advanced |                 132 |                     9,559 |                   100% |               1.00 |                     264 |             9 |

Advanced vs Baseline: 56% fewer prompt tokens processed, recall 0% -> 100%.

