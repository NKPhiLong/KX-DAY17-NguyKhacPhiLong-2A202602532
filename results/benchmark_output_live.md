Mode: live | compact threshold: 600 tokens, keep 4 messages

## Standard Benchmark
conversations.json: 10 conversations, 101 turns, 14 recall questions

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |               6,078 |                    43,309 |                    11% |               0.45 |                       0 |             0 |
| Advanced |              10,453 |                    77,452 |                   100% |               0.99 |                   1,039 |            28 |

Advanced vs Baseline: 79% more prompt tokens processed, recall 11% -> 100%.

## Long-Context Stress Benchmark
advanced_long_context.json: 1 conversations, 16 turns, 3 recall questions

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |               3,356 |                    47,125 |                     0% |               0.40 |                       0 |             0 |
| Advanced |               3,299 |                    19,569 |                   100% |               1.00 |                     793 |            28 |

Advanced vs Baseline: 58% fewer prompt tokens processed, recall 0% -> 100%.

