# Apple M4 Max / 128GB / 40 GPU cores

**Model:** qwen3.5-122b-a10b-mtp  
**Backend:** lmstudio  
**Scenario:** creative-writing (single-shot)  

| Turn | Context | Prefill | Gen | Gen tok/s | Effective tok/s | Total | Output |
|-----:|--------:|--------:|----:|----------:|----------------:|------:|-------:|
| 1 | 57 | 0.63s | 11.95s | 38.3 | **36.4** | 12.57s | 458 |
| 2 | 60 | 0.64s | 17.39s | 37.4 | **36.1** | 18.03s | 651 |
| 3 | 58 | 0.66s | 10.88s | 37.8 | **35.6** | 11.54s | 411 |

**Total prefill:** 1.9s  
**Total generation:** 40.2s  
**Total time:** 42.1s  
**Avg generation tok/s:** 37.8  
**Avg effective tok/s:** 36.1  
