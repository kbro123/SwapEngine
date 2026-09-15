
### 1.3 Timings — PENDING (held for a quiet machine; see status at top)

For orientation only, from the loaded-machine scalar probe of the previous note (`AADJIT-research.md`, load 3.5-11):

| rung | value eval µs, engine | value eval µs, scalar collapsed | value+J µs, engine J alone | value+J µs, scalar collapsed fwd+J |
|---|---|---|---|---|
| ois_nolag | 0.49-0.82 | 0.79-1.34 | 3.0-5.2 | 5.0-8.6 |
| desk | 9.3-11.6 | 19.5-22.7 | 195-264 | 111-116 |
| mixed_scheme | 39-50 | 4.4-5.0 | 93-187 | 23-35 |
| desk_mixed | 141-175 | 20.6-22.7 | 786-1,156 | 111-168 |

The coarse lowering cuts dispatch from ~15k to 122-184 per desk eval (1.2), which removes the loss mechanism that note identified. Whether
it **matches** the hand-written W-cache on linear rungs is exactly what the held timing run measures; it is not claimed here.

