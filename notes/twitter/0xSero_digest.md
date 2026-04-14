# 0xSero Twitter Digest — Local LLMs, REAP, Apple Silicon

Source: `notes/twitter/0xSero_tweets.json` (1,143 tweets, 2026-03-25 → 2026-04-12) + rig sale thread.

Tweets are quoted verbatim. URL format: `https://twitter.com/0xSero/status/<id>`.

---

## 1. About / Background

0xSero is a heavy local-LLM user and model-pruner. American living in Warsaw ("wawa"). Primary focus: publishing REAP-pruned MoE models (Qwen3.5, Gemma-4, Qwen3-Coder-Next, GLM, MiniMax, Arcee Trinity) to Hugging Face. Runs these as a daily driver inside harnesses like Droid (Factory AI, sponsored), Zed, RooCode, OpenCode, Pi. Sponsored by Factory AI / Cerebras. Self-describes as spending most GPU time on pruning, quantizing and observation runs — not on day-to-day inference, where he mainly uses cloud subs (OpenAI, ZAI/GLM, Kimi, MiniMax).

### Hardware fleet (snapshot ~2026-04-09)

**2043063094046339539** (2026-04-11) — the rig he put up for sale in Warsaw:
> I paid 12k usd
> It's
> 512gb ddr4
> 192gb vram (8x 3090s)
> 6tb NVMe
> ASrock romed82t
> Epyc 7443p

**2043063177676562581**: "1200w at peak"
**2043248105164820939**: "150w per gpu / 30% loss in speed / 60% reduction in sound / 70% reduction in cost"

**2042197684904173836** (2026-04-09) — total fleet:
> 288gb VRAM
> - 8x RTX 3090
> - 1x RTX 6000
> Unknown amount of mixed memory expecting 176gb
> 128gb Framework
> 16gb Mac mini
> 32gb Mac M1 Max
> 464gb total memory

**2041887628954329281** (2026-04-08):
> Framework Desktop - arrived
> 1x RTX 6000 - arriving this week
> 3x RTX 6000 - purchasing in a few days
> 8x RTX 3090 - donating to a friends lab
> Intel cards - being donated to me soon
> Mac M5 Ultra - purchasing at launch
> 1TB of memory very soon

**2042363444679086303** (2026-04-09): "Just put in the purchase 3 more 6000s on the way. I have officially hit 384gb VRAM of Blackwell."

**2043108546988913033** (2026-04-11) — what he was pruning that week:
> 1. Framework desktop quantising and benchmarking all my qwen3.5-122B reaps on the strix halo
> 2. 8x 3090s inference for my friend qwen3.5-262b
> 3. 4x B200s observing GLM-5.1 for reap. Should be done Thursday
> 4. 8x H100 observing Qwen3.5-397B for pruning done Friday
> 5. H100 observing Trinity Large Thinking for pruning
> 135M tokens of observation data, packaged into batches of 16k tokens
> - cyber security, philosophy, math, reasoning, coding, agentic, terminal, browser, cuda, my own sessions

---

## 2. REAP Technique

REAP is an expert-pruning method for MoE models. 0xSero collaborates with Cerebras (who wrote the REAP paper) and publishes pruned variants. He recommends readers start with the Cerebras REAP blog post (**2041217540081303884**).

### How REAP works (his explanation)

**2041089432288194743** (2026-04-06):
> The reason they go up is that often only a few of the total experts contribute meaningfully to the model response, but we always light up say 8 experts (when only 3 were needed)
> This adds statistical noise that confuses the model

**2041156055892177099** (2026-04-06):
> I did research on exactly this, about 27% of an LLM is responsible for 95% of all token routing, there's a huge opportunity for compression, problem is predicting the right experts.
> Erwin Bello's research is very interesting.

**2040422132815892663** (2026-04-04):
> It does need them, no doubt about it. We need the model to be smaller more than the model needs experts.

### REAP only works on MoE

**2039773652082635196** (2026-04-02): "Is there a pruning algo for dense models?" — i.e. REAP is MoE-only; dense models are not addressable via this technique.

### Observation data / calibration

**2042366075958878650** (2026-04-09):
> My latest REAPs have 30 million tokens of cyber security samples, 300M tokens total I spanned
> Philosophy, Religion, Math, Coding, Terminal use, Agentic traces, Long reasoning, Cyber security, CUDA, Distributed compute
> + 30M of hand curated sessions from my history

**2042494119591764439**: "That means my REAPs will maintain cyber security performance"

**2039761243800530945** (2026-04-02):
> Keeping to my promise, I need to go over and make sure they can be made public safely. Every run I've done is fully tracked, everything is in a git repo. I am calibrating on 2 years of my own AI sessions so I need to be really careful

### Quality vs speed tradeoff / claims

**2040822269400723955** (2026-04-05, Gemma-4-21B-REAP release):
> Gemma-4-21B-REAP is out! Results are great it held up really well and actually gained accuracy on reasoning tasks.
> MLX & GGUF bros do you thing!
> This should fit on as little as 12GB of vram with some context, or 16GB with full context

**2040826206891782512** (2026-04-05, Qwen3-Coder-Next-REAP):
> Qwen3-Coder-Next-REAP are out! This will fit with full context in 48-62GB of VRAM the results are very solid. I need some MLX and GGUF bros to finish the job and get some compressssions out.
> - 20% … - 30% …

(Note **2040828427922526480**: he swapped the links — first is the 30% variant, second is the 20%.)

**2042586287090073752** (2026-04-10):
> Figuring out this Framework stuff has been so fun, going to port REAP to work with streaming from NVMe + layerwise
> Frameworks could do pretty good REAPs (:
> Might actually be perfect for it.

### Defending REAPs

**2040827064371073375** (2026-04-05):
> "REAPs are broken" "Don't make local AI a disappointment"
> Blah, blah, blah. Making AI accessible to everyone, never stopping, never slowing down.
> AI inference is responsible for the majority of the disappointments people experience. It's not the methodology.

**2042239063416074313** (2026-04-09): "This is an inference issue, the backend I am using to serve this is slopped up. not the models fault. You fix the source of the issue which is my backend lol" — i.e. bugs users hit are upstream inference bugs, not REAP damage.

**2043033393089106013** (2026-04-11): "I'm working on a prune for it" (Arcee)
**2041873227119575045** (2026-04-08): "I'm aware. I'm trying to prune it" (GLM-5.1)

---

## 3. Model Releases (chronological)

| Date | Release | Notes / tweet |
|---|---|---|
| 2026-03-25 | Announcing REAP experiments on Qwen3.5 | **2036603785879962008**: "Testing this tomorrow, will report back if it works on Qwen3.5. Might be able to run much larger models if this works." |
| 2026-03-30 | **Qwen3.5-262B REAP in 4 bits** working on 8x 3090 | **2038758232731119703**: "Droid + Zed make a racing game, 2 shots. … good speed 8x 3090 2x speed" |
| 2026-03-31 | **Qwen3.5-262B-REAP** in vLLM-studio + Factory | **2039104943555035302**: "Qwen3.5-262B-Reap … It can do subagents, swarming, building sites etc.. on 192GB VRAM. It's not production grade code but it is much high quality than anything below 200B" |
| 2026-04-02 | Announces **Gemma-4-26B-REAP in progress** | **2039772353815568555**: "Gemma4-26B-REAP in progress. You will be able to run it on 12GB of VRAM, 16 to be comfortable." |
| 2026-04-02 | 4-bit MLX quant of Gemma-4 REAP (by @ptremblay) | **2039779536070410726** RT + **2039787164607594765**: "Gemma MLX is up 24gb MacBooks singing today" |
| 2026-04-02 | "Apparently it running on 16gb at 40 tokens/s" | **2039789696524382248** |
| 2026-04-05 | **Gemma-4-21B-REAP released** | **2040822269400723955**: "Results are great it held up really well and actually gained accuracy on reasoning tasks. … This should fit on as little as 12GB of vram with some context, or 16GB with full context" |
| 2026-04-05 | **Qwen3-Coder-Next-REAP 20% and 30%** released | **2040826206891782512**: "48-62GB of VRAM … results are very solid" |
| 2026-04-05 | Also noted **Gemma4-19B-REAP** | **2040905388984271018**: "Gemma4-21B and Gemma4-19B REAP results looking really good" |
| 2026-04-05 | **Q4_K_M GGUF REAP of Gemma-4** working | **2040928330627580037**: "10-11gb of vram Q4_K_M reap of Gemma-4 working." |
| 2026-04-08 | Pruning **GLM-5.1** in progress | **2041861620859564512**: "I love GLM-5.1 I am trying to prune it now" |
| 2026-04-09 | **Qwen3.5-262B-REAP-4bit** GPT clone demo | **2042022126203420729** |
| 2026-04-10 | Shares vLLM serve command for **Qwen3.5-REAP-262B-A17B-W4A16** (see §6) | **2042614094801297570** |
| 2026-04-10 | "Some benchmarks." | **2042616159409729564** (image-only; no numbers in text) |
| 2026-04-11 | **Qwen3.5-122B-REAP-q6** numbers on Framework Ryzen 128GB | **2043097446146830374**: "305 tokens/s prefill / 29.2 tokens/s decode / basically can serve 2 users at full context" |
| 2026-04-11 | **Arcee Trinity Large REAP** coming — "will fit on 192gb vram" | **2043098842971635770** |
| 2026-04-11 | Pipeline: GLM-5.1, Qwen3.5-397B, Trinity Large Thinking being observed for pruning | **2043108546988913033** |
| 2026-04-12 | **MiniMax REAP and quants in 7 days** | **2043240465651106282** |

Models he considers his best local ones: **2040100346580676811**: "I'm using it almost exclusively with my inference. Qwen-3.5-262B works perfectly in their harness." And **2039760009572004178**: "No one has been able to knock this one off my GPUs, I am genuinely extremely satisfied with this. Best local model I have so far." (context: Qwen3.5-262B-REAP).

---

## 4. Quality Claims (per pruning ratio)

Explicit quality-per-ratio statements are sparse — 0xSero mostly ships the models with ratios and points at repo benchmark cards rather than tweeting pp-drop numbers.

What he *does* claim directly:

- **Gemma-4-21B-REAP** (ratio not stated as % but 26B → 21B ≈ 19% pruned): **2040822269400723955** — "held up really well and actually **gained accuracy on reasoning tasks**."
- **Gemma-4 19B / 21B**: **2040905388984271018** — "Gemma4-21B and Gemma4-19B REAP results looking really good."
- **Qwen3-Coder-Next-REAP 20%** and **30%** variants: **2040826206891782512** — "results are very solid." Fits "48-62GB of VRAM" with full context. (No MMLU/GPQA numbers posted for these in the tweet stream — he points users to the HF model cards.)
- **Qwen3.5-262B-REAP 4-bit**: **2039104943555035302** — quality "much higher than anything below 200B"; still "not production grade code."
- **262B at 4bit**: **2038758232731119703** — "Overall good performance for a digital assistant that you own, not anywhere close to frontier capabilities."
- General framing (**2039642102154338383**): "Terrible is not the right word, they are inferior to frontier models in most aspects. They are better than frontier 6 months ago in most aspects."

**No tweet in the corpus cites a specific percentage-point MMLU/GPQA/HumanEval drop for any REAP ratio.** Benchmark numbers live in HF model cards and the image at **2042616159409729564** (captioned "Some benchmarks.").

---

## 5. Benchmark Results

Text benchmark numbers in the corpus:

**2043097446146830374** (2026-04-11) — Qwen3.5-122B-REAP-q6 on Framework Ryzen AI 128GB:
> - 305 tokens/s prefill
> - 29.2 tokens/s decode
> - basically can serve 2 users at full context

**2039789696524382248** (2026-04-02) — Gemma-4 REAP (MLX):
> Apparently it running on 16gb at 40 tokens/s

**2040853584246435974** (2026-04-05):
> The model is blazing fast, looks like 250~ tokens/s

**2039645487851024813** (2026-04-02) — generic claim for sglang/vllm + autoround 3-bit:
> you'll get 60-80 tokens/s

**2041086690438045838** (2026-04-06) — on a large rig:
> MiniMax-M2.7 at full precision with 40-100 tok/s with 1 million context
> GLM-5.* at full precision at 15-40 tokens/s with 500k context (total not in 1 session)

**2042239063416074313** — cautions numbers are inference-backend-dependent; perceived quality drops on REAPs are usually backend bugs, not pruning.

**2043105362471956667** (2026-04-11) — Framework Desktop speedups: "Vulkan RADV 29% faster decode".

---

## 6. Inference Stack Opinions

### Framework ranking

**2040136236845682867** (2026-04-03) — his canonical recommendation:
> 3. Use the right frameworks:
> - VLLM, and SGlang for faster inference if you have 1, 2, 4, 8, 16 GPUs
> - exllamav3 and llama.cpp if you have non-power of 2 GPUs
> - MLX if you have Mac

### Against llama.cpp for throughput

**2039645487851024813** (2026-04-02):
> Use sglang/vllm. llama.cpp is a waste of speed, like 3-5x less throughput. Use autoround 3 bit quantisation to get them to fit you'll get 60-80 tokens/s

### llama.cpp chat-template bugs on Gemma-4-REAP

**2040923105393090761** (2026-04-05):
> Seems like a llama.cpp upstream issue, I can't reproduce in bf16, will keep digging. It's a Gemma-4-jinja template issue

**2041055816346718589** (2026-04-06):
> If it still has issue dig around the llama.cpp issues tab, this is a new chat template that isn't well supported. I am running things in vLLM where support moves much faster and I have 0 issues

**2041062149615734937** (2026-04-06):
> These are quantizations of my REAPs for llama.cpp they have issues with the chat template look in my replies on my profile to understand

**2041057418315014233** (2026-04-06): "Exact proof of my point" (pointing at the llama.cpp chat-template issue)

### Turboquant

**2039712891293581532** / **2039714097676091784** (2026-04-02):
> Unfortunately the model weights are the bulk of the VRAM usage. Turboquant also doesn't make vLLM (for example) use less space for cache it just makes more cache fit in the same vram.
> So this is like more free cache instead of less active VRAM for the same context

**2041071413646262427**: "You can get double the context search for the llama.cpp turboquant for Gemma! Going to have to search a bit."

### exllamav3

**2042515781364207650** (2026-04-10):
> Exllamav3 solves this. Also you can do pipeline parallelism over RPC

### MLX on Apple

**2042241025926668697** (2026-04-09):
> on Apple MLX is the best performing engine, unfortunately it's not the fastest.
> Ask your preferred clanker how to increase speed

### vLLM + ROCm

**2043241146764165481** (2026-04-12):
> You can use vLLM with rocm but it only works for bf16 and gguf

### Actual vLLM serve command (his reference config)

**2042614094801297570** (2026-04-10):
```
vllm serve \
  /mnt/llm_models/Qwen3.5-REAP-262B-A17B-W4A16 \
  --served-model-name qwen35-262b \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --trust-remote-code
```

### KV-cache tips

**2039752479097679910** (2026-04-02):
> Quantize kv to 4bit use model in 4bits and you'll get like 20x the kv cache

### Perplexity opinion

**2039666631433932849** (2026-04-02):
> PPL is almost inconsequential for anything other than world knowledge

---

## 7. Hardware / Apple Silicon

### MacBook / MLX

**2039294112092844258** (2026-04-01) — heterogeneous stack thesis:
> Heterogenous hardware is the way forward.
> Large cheap pools of mixed memory + specialized accelerators (Nvidia GPUs, DGX Spark, Cerebras wafers)
> The next year will be dominated by solutions that split the stack.
> - 3000$ for a used Mac Studio

**2039297596422836666** (2026-04-01):
> MacBooks can't stay the size they are if we are going to run fast prefill and decode on it

**2039300781946319263**: "120gb RAM is too slow" (re Apple Silicon pools for bigger models)

**2039787164607594765** (2026-04-02): "Gemma MLX is up 24gb MacBooks singing today"

**2039789696524382248**: "Apparently it running on 16gb at 40 tokens/s" (Gemma-4-REAP MLX).

**2039742489276395818** (2026-04-02) — target hardware list for new SOTA REAP release:
> Do you have ?
> RTX 3090, 4090, 5090
> MacBook Pro 24-96 GB
> DGX spark
> Y'all are eating good today, new local SOTA just for you.

**2042200561282322600** (2026-04-09):
> Once the M5 Ultra is out I'll be picking one up too, I think the future is:
> 1. mixed hardware — 1x Nvidia for decode acceleration / Spark + Framework + Mac for cheap memory
> 2. ASIC type systems. Like Gameboy cartridges for a single model

**2042241025926668697**: "on Apple MLX is the best performing engine, unfortunately it's not the fastest."

### Framework Desktop (Strix Halo / Ryzen AI 128GB)

**2043105362471956667** (2026-04-11):
> how to speed up your framework desktop performance:
> 1. setup 1280GiB GTT
> 2. use MoEs (10B active or less is best)
> 3. Vulkan RADV 29% faster decode
> 4. the smaller the active params the faster it go

**2043097446146830374**: Qwen3.5-122B-REAP-q6 on Framework → 305 tok/s prefill, 29.2 tok/s decode, 2 concurrent users possible at full context.

**2043099370854252845** (2026-04-11) — on Framework/Strix-class machines generally:
> they're incredibly cheap for the amount of memory and the actual all in one cleanliness, memory bandwidth is really low, though. It's slow though, don't expect to run anything over 10B active params at acceptable speeds. It can run a Q4 Reap of MiniMax-m2.5 (soon m2.7) and …

### Mac Mini and M1 Max

**2040572826214695277** (2026-04-04): uses Mac for AI sessions. **2042197684904173836**: "16gb Mac mini, 32gb Mac M1 Max". **2041928660832149911**: "Finally perma switched my Mac Mini to Hermes, it's better."

### On MacBook vs Mac Studio tradeoff

**2043068773163274635** (rig thread, 2026-04-11): "Mac fits larger models but too slow"

### Smaller-active-params principle

**2039841019613548871** (2026-04-02):
> Smaller active params = faster
> Larger active params = deeper intelligence
> Larger total params = wider intelligence
> Very relevant to understand with the new Gemini models.

---

## 8. Recommended Settings

**vLLM reference** (see §6): `--max-model-len 262144`, `--max-num-seqs 8`, `--gpu-memory-utilization 0.9`, `--kv-cache-dtype fp8_e4m3`, `--dtype bfloat16`, `--trust-remote-code`, `--tensor-parallel-size 8`.

**KV cache**: **2039752479097679910** — "Quantize kv to 4bit use model in 4bits and you'll get like 20x the kv cache."

**Quant levels**:
- Qwen3-coder-next on low-VRAM machines (**2041085293810368770**): "Qwen3-coder-next should be the best thing you can run on your hardware, Q6 or if you want try the 62B reap in Q8. It'll be leagues faster and much better imo at agentic and coding"
- Gemma-4-REAP on Q4_K_M GGUF: 10-11 GB VRAM (**2040928330627580037**)
- Qwen3.5 autoround 3-bit: 60-80 tok/s target (**2039645487851024813**)
- Personal daily stack (**2042182401795797370**): "qwen3.5-27B in BF16 / qwen3.5-122B in 4 bits"

**Context**: 262,144 is his standard `--max-model-len` for Qwen3.5-262B REAP.

**Target footprints**:
- Gemma-4-21B-REAP: 12 GB VRAM minimum, 16 GB comfortable for full context (**2040822269400723955**)
- Qwen3-Coder-Next-REAP: 48-62 GB VRAM full context (**2040826206891782512**)
- Qwen3.5-262B-REAP Q4: 192 GB VRAM (**2039104943555035302**)
- Frontier-level open coding in general: "192GB for Q4 quant + REAP" (**2039643150818419032**)

---

## 9. Warnings / Known Issues

### llama.cpp chat-template breakage on Gemma-4-REAP
Repeated warning: the llama.cpp GGUF builds of Gemma-4-REAP have a chat-template (jinja) bug. Not reproducible in bf16 or in vLLM. Users should check llama.cpp issues and prefer vLLM. Tweets: **2040923105393090761**, **2041055816346718589**, **2041057418315014233**, **2041062149615734937**, **2041068002934702149**.

### Inference backend quality bugs ≠ REAP quality problems
**2042239063416074313**: "This is an inference issue, the backend I am using to serve this is slopped up. not the models fault." General caveat that user-reported quality regressions on REAPs are usually backend bugs.

### vLLM ROCm limits
**2043241146764165481**: "You can use vLLM with rocm but it only works for bf16 and gguf".

### llama.cpp throughput
**2039645487851024813**: "3-5x less throughput" vs sglang/vllm.

### Strix Halo / Framework limits
**2043099370854252845**: "don't expect to run anything over 10B active params at acceptable speeds."

### Model-specific caveats
- **GLM-5 (not 5.1) stability**: **2039663343066046771** — "stick it in Droid Missions and watch it flip outputs to Chinese within a few 100k tokens." (He later confirms GLM-5.1 is "a huge leap" and much more consistent: **2041560690234691705**, **2042668286521860314**, **2042684157323571527**.)
- **GGUF support for new templates**: slow to land upstream; vLLM moves faster (**2041055816346718589**).
- **Context length tradeoff on Framework**: turboquant only increases cache fit, doesn't decrease weight VRAM (**2039712891293581532**, **2039714097676091784**).

### Personal confirmation that Qwen3-Coder-Next is best-of-breed on his hardware class
**2042862380745478379**: "Qwen3-coder-next was excellent".

---

## 10. Comparisons to Frontier Models

### His open-weight top-tier list

**2039643150818419032** (2026-04-02) — canonical list:
> Here are all the open weight models that can get close frontier level code, and tie for agentic purposes.
> GLM-5.*
> MiniMax-M2.*
> Kimi-K2.5
> Deepseek-V3.2
> Qwen-3.5-Plus-397B
> If you want AI at home for coding agents similar to Claude/Codex the VRAM needed 192GB for Q4 quant + REAP

**2039642102154338383**:
> Terrible is not the right word, they are inferior to frontier models in most aspects. They are better than frontier 6 months ago in most aspects.
> GLM-5.*, MiniMax-M2.*, Kimi-K2.5, Deepseek-V3.2, Qwen-3.5-Plus-397B

### Task routing heuristics

**2039718957733515594** (2026-04-02):
> For cuda Claude
> For coding GPT
> For frontend Kimi
> For unhealthy relationships grok

**2040461221984395480** (2026-04-04):
> #1 - GPT-5.4
> #2 - GLM-5.1
> #3 - Kimi
> #4 - Claude
> #5 - MiniMax
> #6 - Gemini
> #7 - Local Qwen3.5
> #8 - Composer

### On Qwen3.5-262B-REAP vs frontier

**2038758232731119703**: "Overall good performance for a digital assistant that you own, not anywhere close to frontier capabilities."

**2039104943555035302**: "not production grade code but … much higher quality than anything below 200B"

**2042219595633713509** (2026-04-09):
> Qwen3.5-262B-REAP-4bit cleans my Downloads for me in like 2 minutes. Local AI, even 10B param models can be so much more than we know how to harness right now. Intelligence is a spectrum, me and my REAPs are on it.

### GLM-5.1 vs GPT-5.4 / Opus

**2041861620859564512** (2026-04-08):
> GLM-5.1 in Droid via BYOK is better than GPT-5.4 at porting designs into Figma MCP

**2041901507872780597** (2026-04-08):
> GLM-5.1 is state of the art so far:
> 1. Excellent at programming
> 2. Editing videos
> 3. Browser control
> 4. Subagent spawning
> 5. Decent speeds so far.
> 6. Much cheaper than Opus
> 7. Amazing team
> 8. Good at FFMPEG
> 9. Amazing at security.
> 10. No vision but uses subagents to see.

**2042684157323571527**: "first open model IMO that [runs for hours without needing nudging]"

**2042893821675983041** (2026-04-11) — 20M-token 1-shot reverse-engineering job on GLM-5.1 via Droid. "ZAI will have the first non-western #1 model in almost all benchmarks soon."

### Subscription-tier recommendations

**2039764907906777478** (2026-04-02):
> $10/mo: Opencode go / GLM basic / MiniMax basic / HF Pro
> $20/mo: GPT-Plus / Kimi-Code
> $50/mo: Qwen Code

### Best coding agent overall

**2040526722483785964**: "GPT-5.3-Codex is still the best coding agent, no doubt about it. GPT-5.4 is better at computer use, but doesn't match the sheer autistic power Codex holds."

### Harness ranking for local models

**2040445532171108375** (2026-04-04):
> Best harnesses for local models:
> 1. Droid — daily driver; Qwen3.5 models
> 2. Zed IDE — OpenAI-compatible first class
> 3. Pi Coding Agent — open source, very token efficient, supports vllm + open weight
> 4. RooCode — "Steer mode forces local/dumber models to behave"
> 5. OpenCode
> 6. Parchi

### Benchmarks caveat

**2039658727075106969** (2026-04-02):
> Unfortunately benchmarks only test for highly specific skills that these labs are optimising for (Not talking about benchmaxing). These companies are optimising for Coding, and now agentic trajectories. If you try them to do autoresearch etc.. they don't stand a chance

---

## Appendix — Rig sale thread (2026-04-11, 2043060712134603107)

Selling his 8x 3090 rig in Warsaw:
- 512 GB DDR4, 192 GB VRAM (8x 3090), 6 TB NVMe, ASRock ROMED8-2T, Epyc 7443P
- Paid $12k USD
- 1200W peak
- Power-capped variant at 150W/GPU: 30% speed loss, 60% less noise, 70% less cost
- Selling because "just guzzles too much power to keep scaling"
- Replacement fleet: 4x RTX 6000 Blackwell (384 GB VRAM), Framework Desktop, M5 Ultra on launch

**2043068773163274635**: "Mac fits larger models but too slow" — confirms Mac is memory-first, speed-last.
