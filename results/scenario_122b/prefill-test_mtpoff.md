# Apple M4 Max / 128GB / 40 GPU cores

**Model:** qwen3.5-122b-a10b-mtp  
**Backend:** lmstudio  
**Scenario:** prefill-test (single-shot)  

| Turn | Context | Prefill | Gen | Gen tok/s | Effective tok/s | Total | Output |
|-----:|--------:|--------:|----:|----------:|----------------:|------:|-------:|
| 1 | 655 | 1.52s | 2.88s | 38.2 | **25.0** | 4.40s | 110 |
| 2 | 1,453 | 4.65s | 4.07s | 36.7 | **17.1** | 8.71s | 149 |
| 3 | 3,015 | 9.77s | 4.25s | 35.1 | **10.6** | 14.02s | 149 |
| 4 | 8,496 | 42.84s | 4.49s | 33.2 | **3.1** | 47.32s | 149 |

**Total prefill:** 58.8s  
**Total generation:** 15.7s  
**Total time:** 74.5s  
**Avg generation tok/s:** 35.8  
**Avg effective tok/s:** 7.5  
