# Apple M4 Max / 128GB / 40 GPU cores

**Model:** qwen3.5-122b-a10b-mtp  
**Backend:** lmstudio  
**Scenario:** doc-summary (single-shot)  

| Turn | Context | Prefill | Gen | Gen tok/s | Effective tok/s | Total | Output |
|-----:|--------:|--------:|----:|----------:|----------------:|------:|-------:|
| 1 | 425 | 1.39s | 3.05s | 39.3 | **27.0** | 4.44s | 120 |
| 2 | 612 | 1.30s | 2.32s | 40.0 | **25.6** | 3.63s | 93 |
| 3 | 535 | 1.75s | 3.44s | 39.3 | **26.0** | 5.18s | 135 |
| 4 | 524 | 1.63s | 2.93s | 39.5 | **25.4** | 4.56s | 116 |
| 5 | 1,518 | 4.17s | 2.74s | 39.7 | **15.8** | 6.91s | 109 |

**Total prefill:** 10.2s  
**Total generation:** 14.5s  
**Total time:** 24.7s  
**Avg generation tok/s:** 39.6  
**Avg effective tok/s:** 23.2  
