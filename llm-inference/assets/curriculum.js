/* Inference Atlas — curriculum
   Ordered modules and units. Resource ids reference RESOURCES in library.js. */

window.MODULES = [

/* ==================================================================== 00 */
{
  id: 'm00', num: '00',
  title: 'Ground truth: what a forward pass costs',
  tagline: 'Roofline, FLOPs, KV bytes, and the metrics everything else is measured against.',
  focus: 'Every optimization in the rest of this curriculum is a move on one plot. Quantization attacks bandwidth-bound decode. Continuous batching raises arithmetic intensity toward the compute roof. Speculative decoding spends FLOPs — cheap in the memory-bound regime — to buy back sequential memory reads. Disaggregation exists because prefill and decode want opposite machines. Without the roofline, all of it is trivia.',
  outcome: 'A hand-derived arithmetic intensity, KV-cache-per-token figure, and memory-bound decode latency for one specific model on one specific GPU, checked against a real measurement within 30%.',
  units: [
    {
      id: 'u-forward', tier: 1,
      title: 'The decoder-only forward pass, precisely',
      summary: 'Every tensor, every shape, every byte moved in one token of generation.',
      why: 'You cannot reason about inference cost until you can write down every matrix multiply in a transformer block and its dimensions. Most performance confusion traces back to a fuzzy mental model of which matrices are per-token and which are per-sequence, or to forgetting that the LM head is a `[d_model, vocab]` GEMM that can rival an entire layer.\n\nThis unit is not a tutorial on attention. It is an inventory. By the end you should be able to draw a table of every operation in one block with its FLOP count, its parameter count, and its activation memory, from `d_model`, `n_heads`, `n_kv_heads`, `d_head`, `d_ff`, `n_layers`, and `vocab`.',
      core: [
        'A block is: RMSNorm, QKV projection, RoPE, attention, output projection, residual, RMSNorm, MLP (gate/up/down for SwiGLU), residual. Everything else is bookkeeping.',
        'Per-layer parameter count for a GQA + SwiGLU model: `d*(n_q_heads + 2*n_kv_heads)*d_head` for QKV, `n_q_heads*d_head*d` for the output projection, `3*d*d_ff` for SwiGLU MLP. Norms are negligible in FLOPs but not in kernel-launch count.',
        'The MLP is usually 2/3 of parameters and FLOPs in a modern dense model. Attention projections are ~1/3. Attention *scores* are a separate quadratic term that does not live in the parameter count at all.',
        'The embedding table and LM head each hold `vocab*d` parameters. On small models with large vocabularies (e.g. 128k vocab, d=2048) they can exceed 20% of weights, which distorts the 2N rule.',
        'Only Q, K, V, O, and MLP weights are read from HBM every decode step. The embedding table is a gather — one row — not a GEMM. The LM head is a full GEMM against the vocab and is often the single largest kernel in decode.',
        'RoPE, RMSNorm and residual adds contribute <1% of FLOPs but can contribute 10%+ of decode latency if unfused, because they are pure bandwidth with no reuse.'
      ],
      math: [
        {
          title: 'Per-layer parameters (GQA + SwiGLU)',
          body: `P_attn  = d * (n_q * d_h)          # W_Q
        + d * (n_kv * d_h)         # W_K
        + d * (n_kv * d_h)         # W_V
        + (n_q * d_h) * d          # W_O

P_mlp   = 3 * d * d_ff             # gate, up, down

P_layer = P_attn + P_mlp

P_total = L * P_layer + 2 * V * d  # + embeddings and LM head`,
          note: 'For Llama-3-8B: d=4096, n_q=32, n_kv=8, d_h=128, d_ff=14336, L=32, V=128256. P_attn = 4096*(4096 + 1024 + 1024 + 4096) = 41.9M. P_mlp = 3*4096*14336 = 176.2M. Per layer 218M, times 32 = 6.98B, plus 2*128256*4096 = 1.05B embeddings = 8.03B. The arithmetic closes.'
        },
        {
          title: 'FLOPs per token, decode (batch 1, one new token)',
          body: `# Dense GEMMs: every parameter is used once, multiply-accumulate = 2 FLOPs
F_dense = 2 * P_non_embedding

# Attention against the cache of length S
F_attn  = 2 * L * n_q * d_h * S    # Q @ K^T
        + 2 * L * n_q * d_h * S    # scores @ V
        = 4 * L * n_q * d_h * S

# LM head
F_head  = 2 * V * d

F_token = F_dense + F_attn + F_head`,
          note: 'The famous "2N FLOPs per token" is only the first term. At S = 8192 for Llama-3-8B, F_attn = 4*32*32*128*8192 = 4.3 GFLOP against F_dense = 14.0 GFLOP — attention is already 23% of the arithmetic and 100% of the awkwardness, because it reads the whole KV cache.'
        }
      ],
      pitfalls: [
        'Using "2N FLOPs per token" for a model where embeddings are a large share of N. Use non-embedding parameters.',
        'Forgetting the LM head. `2*V*d` for a 128k vocab and d=4096 is 1.05 GFLOP per token — more than three transformer layers.',
        'Assuming MoE models follow the same rule. For MoE, FLOPs track *active* parameters; memory tracks *total* parameters. These differ by 10-30x.',
        'Counting a multiply-accumulate as one FLOP. The convention in every paper you will read is two.'
      ],
      build: [
        'Write a 60-line Python script that takes a HuggingFace `config.json` and prints: parameter count by component, FLOPs per decode token at a given context length, and KV bytes per token. Validate the parameter count against the published model card to within 1%.',
        'Run it on Llama-3-8B, Llama-3-70B, Qwen3-32B, and a DeepSeek MoE config. Explain why the MoE one needs two separate numbers.'
      ],
      checks: [
        'Why does the attention FLOP term not appear in the parameter count?',
        'For a model with 128k vocabulary and d_model 2048, what fraction of weights is the embedding + LM head, and how does that break the 2N rule?',
        'In decode at batch 1, which single kernel moves the most bytes?'
      ],
      res: ['r-kipply', 'r-attention-paper', 'r-llm-numbers', 'r-modular-basics']
    },

    {
      id: 'u-two-phases', tier: 1,
      title: 'Prefill and decode are two different machines',
      summary: 'One request runs a compute-bound phase and then a bandwidth-bound phase, and they fight.',
      why: 'A single request has two phases with opposite resource profiles. Prefill processes the whole prompt in one parallel pass: it is a big GEMM, arithmetic intensity is high, and it saturates tensor cores. Decode produces one token at a time: the same weights are re-read from HBM for a single token of work, arithmetic intensity is ~1-2 FLOP/byte, and tensor cores idle at single-digit utilization.\n\nEverything downstream follows from this asymmetry. Batching exists to amortize the weight read across many decode tokens. Chunked prefill exists because a long prefill blocks decode. Disaggregation exists because the two phases want different GPUs, or at least different scheduling. If you internalize nothing else from Module 00, internalize that decode does the same amount of memory traffic whether you generate for one user or thirty-two.',
      core: [
        'Prefill: `T` tokens in one pass. FLOPs scale with `T`, bytes moved is roughly the weights once. Arithmetic intensity ≈ `T` (in units of FLOP/byte for the weight read). At T=2048 you are firmly compute bound.',
        'Decode: 1 token per request per step. Bytes moved is the weights once *plus* the whole KV cache for every active sequence. Arithmetic intensity ≈ `batch_size`.',
        'This is why batch size is the single most important serving knob: it is literally the arithmetic intensity of decode.',
        'TTFT is dominated by prefill (queueing + compute). ITL/TPOT is dominated by decode (memory bandwidth). Optimizing one often hurts the other — this trade is the core tension of every scheduler.',
        'MFU during decode at batch 1 on an H100 is typically 0.3-1%. That is not a bug; it is the physics of reading 16 GB to compute 16 GFLOP.',
        'Prefill can be *made* to look like decode (small chunks) and decode can be *made* to look like prefill (large batch, speculative decoding). Most modern scheduler design is about moving requests along that spectrum.'
      ],
      math: [
        {
          title: 'Arithmetic intensity of each phase',
          body: `# Weight-read intensity, ignoring KV for a moment
AI_prefill(T, B) = 2 * P * (B*T) / (P * bytes_per_param)
                 = 2 * B * T / bytes_per_param

AI_decode(B)     = 2 * P * B / (P * bytes_per_param + KV_bytes)
                 ~= 2 * B / bytes_per_param      (when KV << weights)

# BF16, bytes_per_param = 2:
AI_prefill(2048, 1) = 2048 FLOP/byte     -> compute bound
AI_decode(1)        = 1 FLOP/byte        -> memory bound by ~300x on H100`,
          note: 'The H100 machine balance is ~295 FLOP/byte (989 TFLOP/s BF16 ÷ 3.35 TB/s). Decode at batch 1 misses it by two and a half orders of magnitude. Decode at batch 256 in BF16 finally approaches it — which is exactly why continuous batching produced 10-20x throughput gains.'
        },
        {
          title: 'Decode step time, memory-bound regime',
          body: `t_step ~= (P * bytes_per_param + sum_over_requests(S_i) * kv_bytes_per_token)
          / (HBM_bandwidth * efficiency)

# Llama-3-70B, BF16, 4x H100 (TP=4), batch 32, avg context 4096:
weights   = 70e9 * 2                 = 140 GB
kv        = 32 * 4096 * 160 KiB      = 20.0 GiB
bytes     = ~161 GB
bandwidth = 4 * 3.35 TB/s * 0.8      = 10.7 TB/s
t_step    = 161e9 / 10.7e12          = 15.0 ms
tokens/s  = 32 / 0.015               = 2130 tok/s aggregate, 66 tok/s per stream`,
          note: 'This first-order model is usually within 20-30% of a well-tuned engine at moderate batch. When measurement disagrees by more than 2x, the cause is almost always kernel-launch overhead, a bad attention backend, or preemption — not the model.'
        }
      ],
      tables: [{
        title: 'The two phases side by side',
        cols: ['Property', 'Prefill', 'Decode'],
        rows: [
          ['Parallelism', 'All prompt tokens at once', 'One token per sequence'],
          ['Bottleneck', 'Tensor core FLOPs', 'HBM bandwidth'],
          ['Arithmetic intensity', 'Hundreds to thousands', '≈ batch size'],
          ['Typical MFU', '40-60%', '0.5-10%'],
          ['Scales with', 'Prompt length (quadratic in attention)', 'Batch size and context length'],
          ['User metric', 'TTFT', 'ITL / TPOT'],
          ['Wants', 'Big chunks, few interruptions', 'Big batches, short steps'],
          ['Best hardware', 'High FLOP/s, FP8/FP4 tensor cores', 'High HBM bandwidth and capacity']
        ]
      }],
      pitfalls: [
        'Reporting a single "tokens/sec" number without saying whether it is prefill or decode. They differ by 10-100x.',
        'Benchmarking with a 32-token prompt and concluding your service is fast. Short prompts hide all prefill cost.',
        'Assuming decode gets faster on a GPU with more FLOPs. It does not. It gets faster on a GPU with more bandwidth.',
        'Treating "MFU is low" as a bug in decode. Low MFU in decode is correct behaviour; the goal is high *bandwidth* utilization.'
      ],
      build: [
        'Serve any 7-8B model with vLLM. Measure TTFT and ITL at prompt lengths 128, 1024, 8192 with 128 output tokens, batch 1. Plot both. Confirm TTFT grows super-linearly and ITL barely moves.',
        'Repeat at batch 1, 8, 32, 128. Confirm per-stream ITL degrades slowly while aggregate throughput rises nearly linearly, then saturates. Find the knee.'
      ],
      checks: [
        'Why does increasing batch size barely change decode step time until quite large batches?',
        'Your ITL doubled after you raised max_num_seqs. Give two distinct explanations and how you would distinguish them.',
        'You move from H100 (3.35 TB/s, 989 TFLOP/s) to H200 (4.8 TB/s, same FLOPs). Predict the change in TTFT and in ITL.'
      ],
      res: ['r-anyscale-batching', 'r-kipply', 'r-modular-basics', 'r-brrr', 'r-pope-scaling']
    },

    {
      id: 'u-roofline', tier: 1,
      title: 'The roofline model and machine balance',
      summary: 'One plot that explains, predicts, and bounds every optimization in this curriculum.',
      why: 'The roofline model says a kernel achieves `min(peak_FLOPs, arithmetic_intensity * bandwidth)`. That is it. Two lines: a horizontal compute roof and a diagonal bandwidth roof, meeting at the machine balance point. Every kernel is a dot under those lines, and every optimization either moves the dot right (more reuse per byte) or lifts the ceiling (better precision, better hardware).\n\nThe reason to derive this by hand rather than read about it is that the ridge point is a hard number for your specific accelerator, and it tells you immediately whether an idea can possibly help. If decode sits at 1 FLOP/byte and your ridge is 295, then any optimization that only reduces FLOPs — a cheaper activation, a sparser MLP — buys you nothing at all. Only optimizations that reduce bytes or increase reuse matter. That single test kills most bad ideas in ten seconds.',
      core: [
        'Machine balance (ridge point) = peak FLOP/s ÷ peak bytes/s. On an H100 SXM in BF16: 989e12 / 3.35e12 ≈ 295 FLOP/byte.',
        'A kernel left of the ridge is memory bound: performance = AI × bandwidth. Right of the ridge it is compute bound: performance = peak FLOPs.',
        'The ridge moves when you change precision. FP8 doubles the compute roof but leaves bandwidth alone, pushing the ridge to ~590 FLOP/byte — so FP8 *without* weight-size reduction makes the memory-bound problem relatively worse.',
        'Quantizing weights to INT4 does the opposite: it does not raise the roof, it halves the bytes, moving the dot right and up along the diagonal. That is why weight-only quantization is the highest-leverage decode optimization.',
        'There is a third regime the classic roofline hides: overhead-bound, where you are limited by Python, kernel launches, or scheduler CPU time. On small models at batch 1, this is often the real limit. CUDA graphs exist for exactly this.',
        'Attention and GEMM have different intensities and belong at different points on the same plot. Draw them separately or you will draw the wrong conclusion.'
      ],
      math: [
        {
          title: 'Roofline',
          body: `Attainable FLOP/s = min( Peak_FLOPs , AI * Peak_Bandwidth )

AI (arithmetic intensity) = FLOPs performed / bytes moved from HBM

Ridge point AI* = Peak_FLOPs / Peak_Bandwidth

if AI <  AI*  -> memory bound   : reduce bytes, increase reuse
if AI >= AI*  -> compute bound  : reduce FLOPs, use lower precision`,
          note: 'Compute AI* once per accelerator and write it on a sticky note. It is the number you compare everything against.'
        },
        {
          title: 'Critical batch size for decode',
          body: `Decode is compute bound when:
    2 * B  >=  AI* * bytes_per_param

    B* = AI* * bytes_per_param / 2

H100 BF16 weights : B* = 295 * 2 / 2 = 295
H100 FP8  weights : B* = 590 * 1 / 2 = 295   (roof and bytes both change)
H100 INT4 weights : B* = 295 * 0.5 / 2 = 74  (roof unchanged, bytes halved twice)`,
          note: 'These ignore KV cache traffic, which grows with batch and context and pushes B* higher in practice. The real crossover on a long-context workload is often never reached, because KV bytes dominate before you get there. That is the entire motivation for KV quantization and eviction.'
        }
      ],
      pitfalls: [
        'Using peak FLOPs *with sparsity* from a marketing spec sheet. Halve it. Dense is what you get.',
        'Comparing an achieved TFLOP/s number against a peak that assumes a different precision.',
        'Placing a whole model on the roofline as one dot. Decode GEMMs, decode attention, and prefill attention are three very different dots.',
        'Forgetting the overhead-bound regime. If your step time does not change when you halve the model size, you are not bandwidth bound — you are launch bound.'
      ],
      build: [
        'Compute the ridge point for every GPU you have access to, in BF16, FP8 and FP4. Record it in your notes.',
        'Take a real decode step from your service, count bytes moved (weights + KV), count FLOPs, place it on the roofline, and predict the step time. Then measure. Report the ratio.',
        'Use Nsight Compute\'s roofline section on one attention kernel and one MLP GEMM from the same step. Explain why they land where they do.'
      ],
      checks: [
        'Your kernel achieves 40 TFLOP/s on an H100 at AI = 12 FLOP/byte. Is it good? Show the calculation.',
        'Why does FP8 *weights and activations* help decode more than FP8 activations alone?',
        'Give a concrete optimization that moves a kernel right on the roofline without changing the algorithm.'
      ],
      res: ['r-brrr', 'r-roofline-paper', 'r-gpu-mode-l4', 'r-modal-gpu-glossary', 'r-kipply']
    },

    {
      id: 'u-kv-arithmetic', tier: 1,
      title: 'KV cache arithmetic',
      summary: 'The formula that decides your maximum concurrency, and the architectures built to shrink it.',
      why: 'The KV cache is the reason LLM serving is a memory management problem rather than a compute problem. Every active request holds `2 * L * n_kv * d_h * bytes` per token, forever, until it finishes. That number, multiplied by context length and concurrency, is what actually limits how many users a GPU can serve — not FLOPs, not the model size.\n\nOnce you can compute KV bytes per token from a config file in your head, a lot of architecture decisions stop being mysterious. MQA, GQA and MLA are not accuracy innovations; they are KV cache compression schemes, and their entire justification is this formula.',
      core: [
        'KV bytes per token = `2 (K and V) * n_layers * n_kv_heads * d_head * dtype_bytes`. Note it does *not* depend on the number of query heads.',
        'Llama-3-8B in FP16: 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KiB per token. A 128k-token context is 16 GiB of KV for a single request — twice the size of the model in INT4.',
        'MHA (n_kv = n_q) is the expensive baseline. MQA sets n_kv = 1. GQA sets n_kv to a small divisor of n_q, usually 4-8, and is the modern default.',
        'MLA (DeepSeek) compresses K and V into a single low-rank latent of width ~512-576 per token per layer, cached instead of per-head K and V. It shrinks KV by roughly an order of magnitude versus MHA at similar quality, at the cost of a more complicated attention kernel.',
        'Sliding-window attention caps KV per layer at the window size, turning a linear-in-context cost into a constant. Hybrid models interleave a few global layers with many local ones (Gemma, Mistral, GPT-OSS) to keep some long-range capability at bounded cost.',
        'KV cache is *not* the only per-request memory. Add the block table, sampling buffers, and CUDA graph pools. Engines reserve a fraction of memory for this; that fraction is a tunable and getting it wrong causes OOM under load, not at startup.'
      ],
      math: [
        {
          title: 'KV cache size',
          body: `kv_bytes_per_token = 2 * L * n_kv_heads * d_head * dtype_bytes

kv_bytes_per_request = kv_bytes_per_token * seq_len

max_concurrent = (GPU_mem * n_gpus - weights - overhead)
                 / (kv_bytes_per_token * avg_seq_len)`,
          note: 'For MLA, replace `n_kv_heads * d_head` with the latent dimension and drop the factor of 2 if the implementation caches a single joint latent (DeepSeek-V2 caches c_KV of width 512 plus a 64-wide decoupled RoPE key, i.e. 576 per token per layer).'
        },
        {
          title: 'Worked: how many users fit',
          body: `Llama-3-70B, BF16 weights, 8x H100 80GB, FP16 KV

weights      = 70e9 * 2                       = 140.0 GB
total memory = 8 * 80 GiB                     = 687.2 GB
overhead 10% =                                  ~68.7 GB
KV budget    = 687.2 - 140.0 - 68.7           = 478.5 GB

kv/token     = 2 * 80 * 8 * 128 * 2           = 327,680 B = 320 KiB
cacheable    = 478.5e9 / 327680               = 1.46M tokens

@ 8k context : 178 concurrent requests
@ 32k context: 44  concurrent requests
@ 128k ctx   : 11  concurrent requests`,
          note: 'Now switch KV to FP8: the cacheable token count doubles to 2.9M and 128k-context concurrency goes from 11 to 22. That single flag is usually worth more than any kernel optimization you will write.'
        }
      ],
      tables: [{
        title: 'KV cache per token, FP16, common models',
        cols: ['Model', 'L', 'n_kv', 'd_head', 'KV / token', 'KV @ 32k ctx'],
        rows: [
          ['Llama-3-8B (GQA)', '32', '8', '128', '128 KiB', '4.0 GiB'],
          ['Llama-3-70B (GQA)', '80', '8', '128', '320 KiB', '10.0 GiB'],
          ['Llama-2-7B (MHA)', '32', '32', '128', '512 KiB', '16.0 GiB'],
          ['Mistral-7B (GQA, SWA)', '32', '8', '128', '128 KiB', 'capped by window'],
          ['Qwen3-32B (GQA)', '64', '8', '128', '256 KiB', '8.0 GiB'],
          ['DeepSeek-V3 (MLA)', '61', '—', '576 latent', '~68.6 KiB', '~2.1 GiB']
        ],
        note: 'MHA vs GQA is a 4x difference. MHA vs MLA is closer to 8x. This table is the whole argument for why attention architecture changed between 2023 and 2025.'
      }],
      pitfalls: [
        'Using `n_heads` instead of `n_kv_heads`. With GQA this overstates KV by 4-8x and will make you buy GPUs you do not need.',
        'Forgetting that the KV cache for a *preempted* request may be recomputed or swapped, not freed — the memory accounting differs between engines.',
        'Sizing for average context when your traffic is heavy-tailed. p99 context drives OOM, not the mean.',
        'Assuming FP8 KV is free. It costs a small amount of quality on long-context retrieval tasks and should be measured, not assumed.'
      ],
      build: [
        'Extend your config parser to print KV bytes/token and a concurrency table at 4k / 32k / 128k for a GPU you specify.',
        'Set `--kv-cache-dtype fp8` in vLLM and confirm the reported number of GPU blocks roughly doubles. Then run a long-context retrieval eval on both and record the quality delta.',
        'Compute the KV footprint of a DeepSeek MLA model and explain in writing why it is not simply `2*L*n_kv*d_h`.'
      ],
      checks: [
        'Why does KV cache size not depend on the number of query heads?',
        'A model with 32 layers, 8 KV heads and head dim 128 needs 200 concurrent requests at 16k context. How much KV memory in FP16? In FP8?',
        'Under what workload does sliding-window attention give you nothing?'
      ],
      res: ['r-kipply', 'r-gqa-paper', 'r-mqa-paper', 'r-deepseek-v2', 'r-modular-kv-offload', 'r-llm-numbers']
    },

    {
      id: 'u-metrics', tier: 1,
      title: 'Metrics that mean something: TTFT, ITL, TPOT, goodput',
      summary: 'Precise definitions, the relationships between them, and Little\'s Law.',
      why: 'Almost every inference benchmark argument on the internet is two people using the same word for different quantities. "Throughput" can mean output tokens/sec, total tokens/sec including prefill, or requests/sec. "Latency" can mean end-to-end, time to first token, or per-token. Until the definitions are nailed down, no comparison is meaningful.\n\nWorse, the metrics trade against each other along a curve, so a single operating point is never a fair summary. The right output of a benchmark is a curve of latency percentiles against load, plus the point where you stop meeting your SLO. That last quantity — goodput — is what actually matters, and almost nobody reports it.',
      core: [
        'TTFT: request arrival to first streamed token. Includes queueing, scheduling, prefill, and the first sampling step. Queue time is often the majority under load, which is why TTFT explodes non-linearly.',
        'ITL (inter-token latency): gap between consecutive streamed tokens. TPOT (time per output token) is usually defined as `(E2E - TTFT) / (output_tokens - 1)` — the mean of ITL. Report both mean and p99 ITL; ITL jitter is what users perceive as stutter.',
        'E2E latency = TTFT + (output_tokens - 1) × TPOT. Because it scales with output length, comparing E2E across runs with different output lengths is meaningless. Fix the output length.',
        'Throughput: output tokens/sec is the honest serving number. Total tokens/sec (input + output) flatters any system with long prompts. Requests/sec is only meaningful with a fixed request shape.',
        'Goodput: requests/sec that meet *all* SLOs simultaneously (e.g. TTFT p99 < 500 ms AND TPOT p99 < 50 ms). A system can have high throughput and near-zero goodput. This is the number to optimize.',
        'Little\'s Law: `concurrency = arrival_rate × residence_time`. It holds for any stable queueing system, needs no assumptions, and instantly tells you the concurrency a target QPS implies — and therefore the KV cache you need.',
        'Percentiles, not means. LLM latency distributions are heavy-tailed because of preemption and variable output length. A mean hides exactly the behaviour that breaks your SLO.'
      ],
      math: [
        {
          title: 'Definitions',
          body: `TTFT      = t(first_token) - t(request_arrival)
ITL_i     = t(token_i) - t(token_{i-1})
TPOT      = (E2E - TTFT) / (N_out - 1)
E2E       = TTFT + (N_out - 1) * TPOT

Throughput_out = sum(N_out) / wall_clock
Goodput        = requests_meeting_all_SLOs / wall_clock

Little's Law:  L = lambda * W
    L      = average concurrency in system
    lambda = arrival rate (req/s)
    W      = residence time (s) = TTFT + (N_out-1)*TPOT`,
          note: 'Little\'s Law is the bridge from a product requirement to a hardware requirement. "20 req/s, 500 output tokens, 40 tok/s per stream" implies W ≈ 12.8 s and L ≈ 256 concurrent requests. Now go check whether 256 requests\' worth of KV fits.'
        },
        {
          title: 'The latency-throughput curve',
          body: `As offered load rises:
  batch size  ^ ->  aggregate throughput ^  (sublinear, saturating)
  batch size  ^ ->  per-token latency    ^  (slow, then sharp)
  queue depth ^ ->  TTFT                 ^^ (linear, then unbounded)

The knee is where d(throughput)/d(load) collapses while
d(latency)/d(load) explodes. Serve to the left of the knee.
Past saturation, throughput is flat and latency is unbounded:
you are only adding queue.`,
          note: 'A benchmark that reports one point on this curve is reporting one pixel of a graph. Sweep from synchronous (concurrency 1) to saturation and publish the curve.'
        }
      ],
      pitfalls: [
        'Including TCP connect and TLS handshake in TTFT for some clients but not others.',
        'Measuring ITL from a client that buffers SSE chunks. You will measure your HTTP stack, not the engine.',
        'Reporting throughput at a concurrency the system cannot sustain (queue growing without bound). If the queue grows monotonically, the run is invalid.',
        'Comparing engines where one streams token-by-token and the other batches N tokens per SSE event.',
        'Averaging over a run that includes warmup, CUDA graph capture, and the first-request compile.'
      ],
      build: [
        'Write down the exact SLO for a hypothetical product: TTFT p95, TPOT p95, and target QPS. Derive required concurrency with Little\'s Law and the KV memory it implies.',
        'Instrument a client that records per-token arrival timestamps to a file. Compute TTFT, ITL distribution, TPOT, and E2E yourself rather than trusting a tool.',
        'Reproduce the latency-throughput curve for your service and mark the knee and the goodput ceiling.'
      ],
      checks: [
        'Your p50 TTFT is 200 ms and p99 is 9 s. Give three mechanisms that produce this shape.',
        'A system reports 12,000 tokens/sec. What four facts do you need before that number means anything?',
        'Derive the concurrency needed for 50 req/s with 800 output tokens at 30 tok/s per stream and 400 ms TTFT.'
      ],
      res: ['r-modular-benchmarks', 'r-guidellm-intro', 'r-vllm-bench-docs', 'r-distserve', 'r-artificial-analysis']
    },

    {
      id: 'u-hardware', tier: 1,
      title: 'Reading a GPU spec sheet like an engineer',
      summary: 'Which numbers matter for inference, which are marketing, and how to sanity-check both.',
      why: 'Accelerator selection is one of the few inference decisions that is expensive to reverse. It is also the one most often made on the wrong number. For decode, HBM capacity and bandwidth dominate; peak FLOPs barely matter. For prefill-heavy or high-batch workloads, the reverse. For MoE models, capacity and interconnect dominate everything.\n\nSpec sheets make this harder than it needs to be by quoting sparsity-doubled FLOPs, by mixing dense and sparse in the same table, and by quoting NVLink bandwidth bidirectionally. Learn to normalize all of it to dense, unidirectional, per-GPU numbers before comparing anything.',
      core: [
        'Halve any FLOP number that says "with sparsity" or has an asterisk. Structured 2:4 sparsity is not something your serving stack is using.',
        'HBM bandwidth is the decode number. Capacity is the concurrency number. FLOPs is the prefill number. Rank candidates on the axis your workload is bound by.',
        'The ops:byte ratio (peak dense FLOP/s ÷ bandwidth) is the single most useful derived quantity, and it is never printed on the spec sheet.',
        'Achieved bandwidth is typically 75-90% of peak for well-written kernels. Achieved FLOPs (MFU) for prefill is typically 40-60%. Budget with the achieved numbers.',
        'Interconnect matters as soon as TP > 1. NVLink within a node is 5-25x the bandwidth of PCIe; crossing nodes over InfiniBand or Ethernet is another order down. TP across nodes is almost always a mistake for latency-sensitive serving.',
        'FP8 and FP4 tensor core throughput is real but only reachable if your engine, kernels and model checkpoint all support it. Check the software stack before paying for the silicon.',
        'For MoE, total-parameter memory capacity is the gate. A model with 37B active and 671B total needs the memory of a 671B model and the compute of a 37B one.'
      ],
      tables: [{
        title: 'What to optimize for, by workload',
        cols: ['Workload', 'Primary constraint', 'Secondary', 'Buy'],
        rows: [
          ['Chat, short prompts, high concurrency', 'HBM capacity (KV)', 'Bandwidth', 'H200 / MI300X class'],
          ['RAG, long prompts, short outputs', 'Compute (prefill)', 'Capacity for prefix cache', 'H100 / B200 class'],
          ['Agentic, long multi-turn context', 'Capacity + prefix cache hit rate', 'Bandwidth', 'Max memory per GPU'],
          ['Batch/offline generation', 'Aggregate throughput per dollar', 'Nothing else', 'Cheapest FLOP/$ that fits'],
          ['Large MoE (DeepSeek class)', 'Total memory + interconnect', 'Bandwidth', 'Multi-node NVLink domain'],
          ['Small model, latency critical', 'Launch overhead, bandwidth', 'Nothing', 'Any modern GPU + CUDA graphs']
        ]
      }],
      pitfalls: [
        'Comparing an "H100 PCIe" quote against "H100 SXM" numbers. PCIe has less bandwidth (2.0 vs 3.35 TB/s) and lower clocks.',
        'Assuming NVLink bandwidth is per-direction when it is quoted bidirectionally, or vice versa. Halve or double accordingly and state which you used.',
        'Choosing a GPU on price per FLOP for a decode-bound workload.',
        'Ignoring that consumer cards (4090, 5090) have excellent bandwidth-per-dollar but no NVLink, no ECC by default, and licensing restrictions in datacenters.'
      ],
      build: [
        'Build the ops:byte table for every accelerator you might plausibly use. Sort by bandwidth-per-dollar and by capacity-per-dollar. Note that the two orderings differ.',
        'For one target model and SLO, compute the cheapest configuration that meets it on three different accelerators. Show your working; the answer is usually not the fastest chip.'
      ],
      checks: [
        'Why is an H200 often a bigger decode win over H100 than its price difference suggests, despite identical peak FLOPs?',
        'When is a cluster of L40S the right answer, and when is it a trap?',
        'You must serve a 671B-total / 37B-active MoE. What is the binding constraint, and what does that imply about topology?'
      ],
      res: ['r-modal-gpu-glossary', 'r-ml-engineering', 'r-llm-numbers', 'r-h100-whitepaper']
    }
  ]
},

/* ==================================================================== 01 */
{
  id: 'm01', num: '01',
  title: 'Attention kernels and the memory hierarchy',
  tagline: 'FlashAttention, online softmax, decode-time attention, and why kernels are written the way they are.',
  focus: 'Attention is where inference performance is won or lost, because it is the only operation whose cost grows with the conversation. This module goes below the framework: the memory hierarchy that makes tiling necessary, the online softmax trick that makes tiling correct, the decode-specific kernels that look nothing like the training ones, and the architectural variants (MQA/GQA/MLA) that exist purely to shrink what those kernels must read.',
  outcome: 'A Triton attention kernel you wrote, profiled against the library implementation, plus a written explanation of why FlashAttention is exact rather than approximate, derived from the online softmax recurrence.',
  units: [
    {
      id: 'u-memhier', tier: 2,
      title: 'The GPU memory hierarchy and why tiling exists',
      summary: 'Registers, SRAM, HBM: three orders of magnitude of bandwidth, and the tax for crossing them.',
      why: 'FlashAttention is often described as "making attention faster". It does not reduce a single FLOP. What it does is avoid writing an `N×N` matrix to HBM and reading it back. Understanding that requires knowing the actual bandwidth and capacity of each level of the hierarchy, and that the gap between on-chip SRAM and HBM is roughly 10-20x in bandwidth and 10,000x in capacity.\n\nThis is also where the term "IO-aware algorithm" comes from. Once you count HBM accesses instead of FLOPs as the cost model, a large family of "obvious" implementations turns out to be badly suboptimal, and the fixes become obvious in turn: fuse, tile, recompute rather than store.',
      core: [
        'Registers: ~256 KB per SM, effectively infinite bandwidth, private to a thread. Spilling to local memory (which lives in HBM) is a performance cliff.',
        'Shared memory / L1 SRAM: ~228 KB per SM on Hopper, ~20 TB/s aggregate. This is the working set FlashAttention tiles into.',
        'L2: ~50 MB on H100, shared across SMs, a few TB/s. Matters more than people think for KV cache reuse patterns.',
        'HBM: 80-192 GB, 2-8 TB/s. Every byte here is 10-20x more expensive than SRAM and is the currency of decode.',
        'Host DRAM over PCIe Gen5: ~64 GB/s. Three orders of magnitude below HBM. Offloading KV here is a real option but changes the performance regime completely.',
        'The arithmetic intensity of an operation is fixed by the algorithm, but the *achieved* intensity depends on whether you keep intermediates on chip. Fusion is intensity engineering.',
        'Hopper adds TMA (asynchronous bulk copies HBM↔SMEM) and WGMMA (warpgroup-level matmul), which let a kernel overlap data movement with compute instead of alternating. FlashAttention-3 is essentially the exploitation of these two features.'
      ],
      math: [{
        title: 'Naive vs IO-aware attention, HBM traffic',
        body: `Naive attention (materialize S = QK^T):
  read  Q, K            : 2 * N * d * b
  write S               : N^2 * b
  read  S, write P      : 2 * N^2 * b
  read  P, V, write O   : N^2*b + N*d*b + N*d*b
  total ~ O(N^2 + N*d)

FlashAttention (tile, never materialize S):
  read Q, K, V once (per tile pass), write O once
  total ~ O(N*d + N^2*d / M)     M = SRAM tile capacity

At N=8192, d=128, b=2 bytes:
  naive S alone   = 8192^2 * 2 = 134 MB written AND read, per head, per layer
  FA2             = never leaves SRAM`,
        note: 'This is why FlashAttention is a 2-4x wall-clock win with zero change in FLOPs and bit-comparable outputs up to reassociation. It is a memory-traffic optimization, not an arithmetic one.'
      }],
      pitfalls: [
        'Assuming shared memory is free. It is banked; conflicting access patterns serialize and can cost more than the HBM read you avoided.',
        'Optimizing occupancy as an end in itself. High occupancy with register spilling is slower than low occupancy without it.',
        'Ignoring L2. On decode with shared prefixes, L2 residency of hot KV blocks is a measurable effect.',
        'Believing FlashAttention helps decode at batch 1. At sequence length 1, there is no `N×N` matrix to avoid; the decode kernel is a different algorithm with different goals.'
      ],
      build: [
        'Write a fused RMSNorm + residual kernel in Triton. Measure achieved bandwidth against `torch.nn.functional` unfused. Explain the gap in bytes, not in vibes.',
        'Use Nsight Compute on one attention kernel and read off: achieved DRAM throughput, L2 hit rate, and SM occupancy. Write one paragraph on which is the limiter.'
      ],
      checks: [
        'Why does fusing two elementwise ops help more than fusing two GEMMs?',
        'What is the approximate HBM traffic saved by FlashAttention for one layer at N=4096, d_head=128, 32 heads, BF16?',
        'Why is "recompute in the backward pass" a *win* despite doing more FLOPs?'
      ],
      res: ['r-brrr', 'r-gpu-mode-l4', 'r-flashattn1', 'r-modal-gpu-glossary', 'r-pmpp']
    },

    {
      id: 'u-online-softmax', tier: 2,
      title: 'Online softmax: the trick that makes tiling exact',
      summary: 'A one-pass, numerically stable softmax recurrence — the mathematical core of FlashAttention.',
      why: 'Softmax needs a global maximum and a global sum for numerical stability, which appears to require seeing the whole row before you can normalize anything. That apparent requirement is what would force materializing the full attention matrix. Online softmax removes it with a rescaling recurrence: process a tile, track the running max and running denominator, and retroactively correct the accumulated output by the ratio of old to new max.\n\nDerive this yourself once. It is four lines of algebra, it is the reason FlashAttention is *exact* rather than approximate, and the same recurrence reappears in FlashDecoding, in split-KV attention, and in every distributed attention scheme (Ring Attention, context parallelism) where partial results from different devices must be combined.',
      core: [
        'Safe softmax subtracts the row max before exponentiating to avoid overflow: `exp(x_i - m)`.',
        'Online softmax maintains `(m, l, o)` — running max, running sum of exponentials, running weighted output — and updates all three per tile.',
        'The correction factor is `exp(m_old - m_new)`. Applying it to both `l` and `o` keeps the invariant exact at every step.',
        'Because the update is associative, partial results from different tiles, different thread blocks, or different GPUs can be merged in any order. This is what makes split-KV and Ring Attention possible.',
        'The output is exact up to floating-point reassociation. Bit-differences versus a naive implementation are expected and are not a bug.',
        'The same merge operation is what FlashDecoding uses to split one query\'s attention across many SMs, which is essential when batch is small and there is not enough parallelism otherwise.'
      ],
      math: [{
        title: 'The recurrence',
        body: `Given tiles of scores s^(1), s^(2), ... for one query row:

  m_new = max(m_old, max(s))
  l_new = l_old * exp(m_old - m_new) + sum(exp(s - m_new))
  o_new = o_old * exp(m_old - m_new) + sum(exp(s - m_new) * V_tile)

Final output:
  O = o_final / l_final

Merging two independent partials (a, b) — used by split-KV / Ring:
  m = max(m_a, m_b)
  l = l_a*exp(m_a - m) + l_b*exp(m_b - m)
  o = o_a*exp(m_a - m) + o_b*exp(m_b - m)`,
        note: 'Milakov & Gimelshein published the online softmax formulation in 2018; Rabe & Staats showed attention needs only O(log n) memory; FlashAttention combined both with tiling and an IO cost model. Read them in that order and the paper stops looking magical.'
      }],
      code: [{
        title: 'Online softmax, 20 lines of NumPy — verify it against the naive version',
        body: `import numpy as np

def online_attention(Q, K, V, tile=64):
    n, d = K.shape
    m = -np.inf; l = 0.0; o = np.zeros(d)
    for i in range(0, n, tile):
        s = Q @ K[i:i+tile].T / np.sqrt(d)
        m_new = max(m, s.max())
        corr  = np.exp(m - m_new) if m > -np.inf else 0.0
        p     = np.exp(s - m_new)
        l     = l * corr + p.sum()
        o     = o * corr + p @ V[i:i+tile]
        m     = m_new
    return o / l

Q = np.random.randn(128); K = np.random.randn(1024,128); V = np.random.randn(1024,128)
ref = (lambda s: (np.exp(s-s.max())/np.exp(s-s.max()).sum()) @ V)(Q @ K.T / 128**0.5)
print(np.abs(online_attention(Q,K,V) - ref).max())   # ~1e-15`,
        note: 'If you cannot write this from memory, you do not yet understand FlashAttention. Type it out.'
      }],
      pitfalls: [
        'Forgetting to rescale `o` as well as `l` when the max updates. The result will be subtly wrong only on inputs where the max moves late.',
        'Initializing `m = 0` instead of `-inf`, which silently breaks the first tile for all-negative scores.',
        'Assuming the merge is only for tiles within one kernel. It is the same operation used across GPUs.',
        'Doing the exponent in FP16. Keep the softmax accumulator in FP32 or you will lose accuracy on long sequences.'
      ],
      build: [
        'Implement the NumPy version above from scratch without looking, and verify max absolute error < 1e-12 against naive softmax attention.',
        'Extend it to merge two independently computed partials, simulating split-KV. Verify exactness.'
      ],
      checks: [
        'Why is FlashAttention called exact when its output differs in the last bits from a naive implementation?',
        'Show that the merge operation is associative and commutative. Why does that matter for Ring Attention?',
        'Where does the `exp(m_old - m_new)` factor come from algebraically?'
      ],
      res: ['r-online-softmax', 'r-rabe-staats', 'r-flashattn1', 'r-gpu-mode-l12']
    },

    {
      id: 'u-flashattention', tier: 2,
      title: 'FlashAttention 1, 2, 3 and what changed each time',
      summary: 'Tiling, then work partitioning, then Hopper asynchrony — three different bottlenecks, three different fixes.',
      why: 'FlashAttention is three papers, and treating them as one blurs the lesson. FA1 identified HBM traffic as the cost model and tiled to eliminate it. FA2 found that FA1 was leaving parallelism and non-matmul FLOPs on the table, and restructured the loop order and work partitioning to fix it. FA3 targeted Hopper specifically, exploiting asynchronous copies and warp specialization so that data movement and tensor core work overlap rather than alternate, plus FP8 with incoherent processing.\n\nEach step is a different kind of optimization: algorithmic (IO-aware), scheduling (parallelism and work division), and hardware-specific (asynchrony and precision). That taxonomy generalizes to every kernel you will ever optimize.',
      core: [
        'FA1: tile Q, K, V into SRAM; use online softmax; never materialize the score matrix. Result: O(N) memory instead of O(N²), 2-4x speedup, exact.',
        'FA2: swap the loop order so the outer loop is over Q blocks; parallelize over sequence length as well as batch and heads; reduce non-matmul FLOPs (the rescaling) which run ~16x slower than matmul on tensor cores. Result: ~2x over FA1, ~70% of peak on A100.',
        'FA3: producer/consumer warp specialization with TMA for async copies, WGMMA for warpgroup matmuls, pingpong scheduling to overlap softmax of one block with the matmul of another, and FP8 with incoherent processing to control outlier error. ~75% of H100 peak, 1.5-2x over FA2.',
        'The special-function unit that computes `exp` is roughly two orders of magnitude slower than the tensor cores on H100. Hiding softmax behind matmul is therefore the central scheduling problem in FA3.',
        'Causal masking is not free: it halves the useful work but naive implementations still schedule the full grid. Good kernels skip fully-masked blocks, which is why causal attention is ~2x faster only in well-written kernels.',
        'Variable-length batching (`varlen`) packs sequences without padding using cumulative-length offsets. Padding-based batching wastes exactly the fraction of tokens that is padding, which on real traffic is often 30-50%.'
      ],
      tables: [{
        title: 'The FlashAttention line',
        cols: ['Version', 'Target', 'Key mechanism', 'Reported result'],
        rows: [
          ['FA1 (2022)', 'HBM traffic', 'Tiling + online softmax + recompute', '2-4x, O(N) memory'],
          ['FA2 (2023)', 'Parallelism, non-matmul FLOPs', 'Loop reorder, seq-len parallelism, better partitioning', '~2x over FA1, ~70% A100 peak'],
          ['FA3 (2024)', 'Hopper asynchrony, FP8', 'TMA + WGMMA, warp specialization, pingpong, incoherent processing', '~1.5-2x over FA2, ~75% H100 peak; FP8 ≈1.2 PFLOP/s'],
          ['FlashDecoding', 'Small batch decode', 'Split KV across SMs, merge with online softmax', 'Large speedup at low batch, long context'],
          ['FlashInfer', 'Serving integration', 'Block-sparse/paged KV formats, JIT kernels, MLA & cascade support', 'Backend for vLLM and SGLang']
        ]
      }],
      pitfalls: [
        'Benchmarking FA against a naive PyTorch attention that materializes scores in FP32. You will measure a number that flatters FA and teaches nothing.',
        'Assuming the FA kernel used in your serving engine is the one from the paper. Engines use FlashInfer, FA2, FA3, cuDNN or Triton backends depending on GPU, dtype and paging. Check which one is selected at runtime.',
        'Using a head dim the kernel does not specialize for. Performance falls off a cliff outside the supported set (typically 64, 128, 256).',
        'Expecting FA3 gains on Ampere. WGMMA and TMA do not exist there.'
      ],
      build: [
        'Write a tiled attention forward kernel in Triton for a fixed head dim. Benchmark against `torch.nn.functional.scaled_dot_product_attention` at N = 512, 2048, 8192. Expect to lose; report the gap and the reason.',
        'Toggle your engine\'s attention backend (e.g. vLLM `VLLM_ATTENTION_BACKEND`) across FlashAttention, FlashInfer and Triton. Measure TTFT and ITL for each on the same workload.'
      ],
      checks: [
        'FA1 and FA2 do the same FLOPs. Why is FA2 twice as fast?',
        'Why does warp specialization help on Hopper but not on Ampere?',
        'What breaks if you apply the FA training kernel unmodified to decode with a paged KV cache?'
      ],
      res: ['r-flashattn1', 'r-flashattn2', 'r-flashattn3', 'r-flashattn3-blog', 'r-gpu-mode-l12', 'r-gpu-mode-l36', 'r-flashinfer']
    },

    {
      id: 'u-decode-attention', tier: 2,
      title: 'Decode-time attention: FlashDecoding, split-KV, paged kernels',
      summary: 'At sequence length one, every assumption behind training kernels breaks.',
      why: 'Training and prefill attention have thousands of query positions to parallelize over. Decode has one per sequence. If your batch is 8 and you have 132 SMs, a naive decode kernel leaves 90% of the GPU idle while it serially walks a 32k-token KV cache. The fix is to parallelize along the *key* dimension instead — split the KV cache into chunks, compute partial attention on each, and merge with the online softmax combine.\n\nOn top of that, production KV caches are not contiguous. They are paged, so the kernel must gather blocks through an indirection table, which changes the memory access pattern and forbids many of the tricks a contiguous kernel would use. Decode attention is therefore a genuinely different kernel family, and it is where most serving engines spend their kernel engineering effort.',
      core: [
        'FlashDecoding splits the KV sequence across thread blocks, computes partial `(m, l, o)` per split, and merges. It converts a low-parallelism problem into a high-parallelism one at the cost of a small reduction pass.',
        'The optimal number of splits depends on batch × heads versus SM count. Engines pick it heuristically at runtime; a bad heuristic is a common source of long-context decode slowness.',
        'Paged KV kernels take a block table and gather 16- or 32-token blocks. Gather adds indexing overhead and hurts coalescing versus contiguous reads — this is the real cost of PagedAttention, and it is small but not zero.',
        'GQA lets the kernel load one KV head and reuse it across several query heads from registers/SMEM, which raises decode arithmetic intensity by the group size. GQA is a bandwidth optimization *at the kernel level*, not just a memory-capacity one.',
        'MLA decode kernels (FlashMLA, FlashInfer MLA) must decompress the latent, which converts some memory traffic into compute — favourable in the memory-bound regime, and a good example of the roofline logic in action.',
        'Cascade / shared-prefix attention computes the shared prefix once for a whole batch and merges it with per-request suffix attention using the same combine operation. On a system prompt shared by every request, this is a large win.'
      ],
      math: [{
        title: 'Why split-KV is necessary',
        body: `Parallel work available in decode attention without splitting:
    blocks = batch * n_q_heads

H100 has 132 SMs. With batch=4, n_q_heads=32:
    blocks = 128  -> barely one wave, no latency hiding

Split KV into S chunks:
    blocks = batch * n_q_heads * S
    each chunk does (kv_len / S) work, then a cheap merge

Choose S so that batch * heads * S ~= 2-4x SM count,
and kv_len/S stays large enough to amortize the merge.`,
        note: 'This is the decode analogue of FA2\'s "parallelize over sequence length" insight, applied to keys instead of queries.'
      }],
      pitfalls: [
        'Using a prefill kernel for decode. It will work and it will be slow, and the profile will show terrible occupancy rather than an obvious error.',
        'Assuming GQA gives no kernel benefit because the FLOPs are unchanged. The benefit is in KV bytes read per query head.',
        'Ignoring block size. Small pages reduce fragmentation but increase gather overhead; 16 is a common compromise, and it is a tunable.',
        'Measuring decode attention in isolation. It is interleaved with GEMMs, and the interaction (L2 pressure, launch gaps) is where real time goes.'
      ],
      build: [
        'Sweep KV cache block size in your engine (e.g. 8, 16, 32) at long context and measure ITL. Explain the shape of the curve.',
        'Construct a workload where every request shares a 4000-token system prompt. Measure TTFT with and without prefix caching, and separately with a cascade-attention-capable backend if available.'
      ],
      checks: [
        'At batch 1 and 128k context, what limits decode attention throughput, and how does split-KV change it?',
        'Why does a paged KV layout cost performance relative to contiguous, and why is it worth it anyway?',
        'How does GQA change the arithmetic intensity of the decode attention kernel specifically?'
      ],
      res: ['r-flashdecoding', 'r-flashinfer', 'r-vllm-paper', 'r-gpu-mode-l35', 'r-vattention']
    },

    {
      id: 'u-attn-variants', tier: 2,
      title: 'MQA, GQA, MLA and sliding windows: architecture as KV compression',
      summary: 'Four architectural choices whose only purpose is to make inference cheaper.',
      why: 'Between 2022 and 2025 the attention block changed several times, and every change was driven by inference economics rather than by quality. Multi-query attention traded quality for an 8-32x KV reduction. Grouped-query attention recovered most of that quality at 4-8x reduction and became the default. Multi-head latent attention compressed K and V into a shared low-rank latent, getting MHA-like quality at better-than-MQA memory. Sliding windows bounded the cost entirely at the price of range.\n\nKnowing *why* each was adopted lets you predict how a new model will behave in production before you serve it, and lets you read a config file and immediately know your concurrency ceiling.',
      core: [
        'MQA (Shazeer 2019): one KV head shared by all query heads. Maximum compression, measurable quality loss, and training instability at scale. Used by PaLM, Falcon.',
        'GQA (Ainslie 2023): `g` KV heads, each shared by `n_q/g` query heads. Interpolates between MHA and MQA; `g = 8` is the industry default. Can be produced by uptraining an MHA checkpoint with ~5% of original compute.',
        'MLA (DeepSeek-V2/V3): project K and V down to a shared latent `c_KV` (e.g. 512 dims) cached per token per layer, plus a small decoupled RoPE key. Up-projection is absorbed into the query and output matrices so it costs compute, not cache. Roughly 8-14x smaller KV than MHA with quality that matches or beats it.',
        'Sliding-window attention: each token attends to the last `w` tokens. KV per layer is capped at `w`, making memory independent of context length. Requires either global layers interleaved or attention sinks to avoid quality collapse.',
        'Hybrid stacks (a few global layers, many local ones) are now common precisely because pure local loses long-range retrieval and pure global is unaffordable.',
        'Every one of these changes the decode kernel. Engine support lags model releases; "supported" and "supported with a fast kernel" are different states.'
      ],
      tables: [{
        title: 'KV per token per layer, d_model 4096, 32 query heads, d_head 128, FP16',
        cols: ['Scheme', 'Cached width', 'Bytes/token/layer', 'Relative'],
        rows: [
          ['MHA (32 KV heads)', '2 × 32 × 128', '16,384 B', '1.0x'],
          ['GQA g=8', '2 × 8 × 128', '4,096 B', '0.25x'],
          ['GQA g=4', '2 × 4 × 128', '2,048 B', '0.125x'],
          ['MQA g=1', '2 × 1 × 128', '512 B', '0.031x'],
          ['MLA (latent 512 + rope 64)', '576', '1,152 B', '0.070x'],
          ['SWA w=4096 (MHA heads)', 'min(S, 4096) rows', 'capped', 'bounded']
        ],
        note: 'MLA achieves near-MQA memory with MHA-class quality, which is why it spread quickly. The cost is kernel complexity: the up-projection has to be fused into surrounding matrices or you lose the win.'
      }],
      pitfalls: [
        'Setting tensor parallel degree greater than `n_kv_heads`. KV heads must then be replicated across ranks, and your KV memory stops shrinking with TP — a very common and very expensive surprise on GQA models.',
        'Assuming MLA is a drop-in win. Without a fused/absorbed implementation, the up-projection cost can exceed the bandwidth saved at small batch.',
        'Believing sliding-window models cannot do long-range retrieval at all. Hybrid stacks usually can; pure-local ones usually cannot. Test, do not assume.',
        'Comparing KV sizes across models without normalizing for layer count. A 61-layer model with small KV per layer may still cache more than a 32-layer model with large KV.'
      ],
      build: [
        'For four models you might serve, compute KV bytes/token, and the maximum concurrency at 32k context on one H100. Rank them. The ranking will not match parameter count.',
        'On a GQA model with 8 KV heads, run TP=4 and TP=16 and observe how per-GPU KV memory changes. Explain the discontinuity.'
      ],
      checks: [
        'Why does GQA with `g=8` under TP=16 waste KV memory, and what are your options?',
        'Explain how MLA avoids paying the up-projection cost at every decode step.',
        'Why do attention sinks matter for sliding-window and streaming inference?'
      ],
      res: ['r-mqa-paper', 'r-gqa-paper', 'r-deepseek-v2', 'r-deepseek-v3', 'r-streamingllm', 'r-mistral-swa']
    },

    {
      id: 'u-overhead', tier: 2,
      title: 'The overhead-bound regime: launch cost, CUDA graphs, torch.compile',
      summary: 'When the GPU is idle and the profiler is empty, the bottleneck is your host code.',
      why: 'The roofline has two roofs; production has three regimes. On small models, at small batch, with many small kernels, neither compute nor bandwidth is the limit — the limit is the CPU issuing work. A 8B model decode step can involve 300+ kernel launches, and at ~5-10 µs of launch and Python overhead each, that is milliseconds of pure gap before any math happens.\n\nThis is why CUDA graphs exist, why vLLM V1 restructured its scheduler to reduce CPU work per step, and why SGLang made "zero-overhead scheduler" a headline feature. It is also the single most common reason a hand-rolled serving loop is 3x slower than an engine, and it will not show up as low bandwidth or low FLOPs — it shows up as gaps in the timeline.',
      core: [
        'Kernel launch overhead is ~3-10 µs. A decode step with 300 kernels and no graph capture spends 1-3 ms on launches alone, which at 30 ms/step is tolerable and at 5 ms/step is fatal.',
        'CUDA graphs capture a static sequence of kernels and replay it with one launch. They require static shapes, which is why engines capture graphs for a set of batch sizes and pad to the nearest captured size.',
        'Graph capture costs startup time and memory (a pool per captured shape). Engines expose the capture list as a tunable; capturing too many shapes wastes memory, too few causes padding waste.',
        '`torch.compile` fuses elementwise chains and reduces launch count; combined with graphs it addresses both overhead and bandwidth. In serving it is used for the model forward, not the scheduler.',
        'The scheduler itself is host code. If the Python scheduler takes 2 ms per step and the GPU step is 8 ms, you have a 25% tax that no kernel work can remove. Overlapping CPU scheduling with GPU execution is the fix, and it is what "zero-overhead scheduler" means.',
        'Diagnose with a timeline profile, not with counters. Gaps between kernels are the signature; low SM occupancy alone is ambiguous.'
      ],
      math: [{
        title: 'When overhead dominates',
        body: `t_step = max( t_gpu , t_cpu ) + t_gaps        (if overlapped)
       = t_gpu + t_cpu + t_gaps                  (if serialized)

t_launch_total = n_kernels * launch_us

8B model, 32 layers, ~10 kernels/layer + head = ~330 kernels
330 * 6us = 2.0 ms of launch overhead

Memory-bound step time at batch 1, H100, BF16:
16 GB / 3.35 TB/s = 4.8 ms

-> overhead is 29% of the step. With CUDA graphs: ~0.05 ms (1%).`,
        note: 'The smaller the model and the lower the batch, the larger this fraction. On a 1B model it can exceed 100% of the compute time.'
      }],
      pitfalls: [
        'Benchmarking without warmup. The first iterations include autotuning, graph capture and JIT compilation, and can be 10-100x slower.',
        'Enabling CUDA graphs and then using a dynamic shape that falls outside the captured set every step, silently disabling them.',
        'Assuming `torch.compile` is free. Compilation can take minutes and recompiles on shape changes; in serving you want a bounded set of shapes.',
        'Profiling with the profiler\'s own overhead included. Cross-check wall-clock with and without.'
      ],
      build: [
        'Run your engine with and without CUDA graphs (`--enforce-eager` in vLLM disables them). Measure ITL at batch 1, 8, 64. Plot the overhead fraction against batch size.',
        'Capture a PyTorch profiler trace of ten decode steps, open it in a trace viewer, and measure total gap time between kernels. Report it as a percentage of the step.'
      ],
      checks: [
        'Why does CUDA graph capture require static shapes, and how do engines cope with variable batch size?',
        'You halve your model size and step time does not change. What is the diagnosis and the next measurement?',
        'What does "zero-overhead scheduler" actually overlap, and what does it not fix?'
      ],
      res: ['r-brrr', 'r-gpt-fast', 'r-vllm-v1-blog', 'r-sglang-v04', 'r-pytorch-compile-docs']
    }
  ]
},

/* ==================================================================== 02 */
{
  id: 'm02', num: '02',
  title: 'KV cache management',
  tagline: 'Paging, prefix reuse, eviction, quantization, and offload — the memory system of an LLM server.',
  focus: 'Serving an LLM is mostly an exercise in managing one large, fragmented, per-request cache whose lifetime you cannot predict. This module treats the KV cache as an operating-systems problem, which is exactly how the field solved it: virtual memory (PagedAttention), a page cache (prefix caching), a replacement policy (LRU, H2O, SnapKV), compression (KV quantization), and a memory hierarchy (GPU→CPU→NVMe).',
  outcome: 'A measured prefix-cache hit rate on a realistic multi-turn workload, a memory-waste figure for your paging configuration, and a quantified quality/memory trade curve for FP8 KV and for one eviction policy.',
  units: [
    {
      id: 'u-paged-attention', tier: 2,
      title: 'PagedAttention and the block manager',
      summary: 'Virtual memory for the KV cache: block tables, copy-on-write, and why waste falls below 4%.',
      why: 'Before vLLM, engines pre-allocated a contiguous KV buffer per request sized to the maximum possible output length. Because you cannot know how long a generation will be, that allocation is almost always far too large: the vLLM paper measured 60-80% of KV memory wasted to internal fragmentation and over-reservation. Since KV memory is what caps concurrency, wasting 70% of it means serving roughly a third of the users you could.\n\nPagedAttention applies the operating-system solution: fixed-size blocks, a per-request block table mapping logical to physical blocks, allocation on demand, and sharing via reference counts with copy-on-write. Waste drops to less than 4% — bounded by internal fragmentation in the last partial block — and sharing becomes possible for free. Read the block manager source, not just the paper; the paper explains the idea and the code explains the engineering.',
      core: [
        'Physical KV memory is carved into fixed-size blocks holding `block_size` tokens (commonly 16) for one layer\'s K and V.',
        'Each sequence has a block table: an array mapping logical block index to physical block id. The attention kernel gathers through it.',
        'Blocks are allocated on demand as generation proceeds. Waste is bounded by `block_size - 1` tokens per sequence, hence <4% for typical block sizes and sequence lengths.',
        'Reference-counted sharing lets parallel samples, beam search branches, and identical prefixes point at the same physical blocks. Copy-on-write triggers only when a shared block is written.',
        'Preemption is now possible and cheap to reason about: evict a sequence\'s blocks and either recompute its KV later or swap it to CPU. Both strategies exist and have different cost profiles.',
        'The trade is a gather-based attention kernel and a block table lookup on every access. It is a small constant cost that buys a large capacity win — a good example of the general pattern that indirection is cheaper than fragmentation.',
        'The V1 engine restructured this further: block hashes for automatic prefix caching, a flat scheduler, and reduced per-step CPU work. The concepts are the same; the code is cleaner.'
      ],
      math: [{
        title: 'Fragmentation, before and after',
        body: `Contiguous pre-allocation:
    reserved = max_seq_len per request
    used     = actual_len
    waste    = (max_seq_len - actual_len) / max_seq_len

    max_len 4096, actual 300  ->  93% wasted

Paged:
    blocks   = ceil(actual_len / block_size)
    reserved = blocks * block_size
    waste    = (reserved - actual_len) / reserved <= (block_size-1)/reserved

    block_size 16, actual 300 -> 19 blocks = 304 tokens -> 1.3% wasted`,
        note: 'Everything else about vLLM follows from having recovered that memory: bigger batches, continuous batching that actually fills the GPU, and prefix sharing.'
      }],
      code: [{
        title: 'What to read in the vLLM source (V1)',
        body: `vllm/v1/core/
  kv_cache_manager.py      # allocation, free, prefix-cache lookup
  block_pool.py            # free list, ref counts, block hashing
  kv_cache_utils.py        # block hash computation, layout
  sched/scheduler.py       # admission, budget, preemption

Read in this order and answer:
  1. What exactly is hashed to identify a reusable block?
  2. When is a block freed vs kept for future cache hits?
  3. What happens to a running request when memory runs out?
  4. Where is the token budget for one step decided?`,
        note: 'Do not reimplement PagedAttention. Read it until you can answer those four questions from memory. That is the deliverable.'
      }],
      pitfalls: [
        'Tuning `gpu_memory_utilization` up to 0.99 to get more blocks, then OOMing under load because activation and CUDA graph pools were not accounted for.',
        'Assuming block size only affects fragmentation. It also affects kernel efficiency and prefix-cache granularity — you can only share on block boundaries.',
        'Confusing preemption-by-recompute with preemption-by-swap. Recompute burns prefill FLOPs; swap burns PCIe bandwidth. Which is better depends on context length.',
        'Believing "<4% waste" applies to your workload without checking. Very short sequences with a large block size can waste much more.'
      ],
      build: [
        'Read the four files above and write a one-page explanation of the block lifecycle from allocation to reuse, citing function names.',
        'Instrument or read `num_preempted_reqs` and KV cache utilization from your engine\'s metrics under a load test that deliberately exceeds capacity. Describe the failure mode you observe.'
      ],
      checks: [
        'Where does the remaining <4% of waste come from, exactly?',
        'Why does copy-on-write matter for parallel sampling but not for single-sample chat?',
        'Your engine reports 95% KV utilization and rising preemptions. Name three levers and their side effects.'
      ],
      res: ['r-vllm-paper', 'r-vllm-blog', 'r-vllm-repo', 'r-modular-paged', 'r-vllm-v1-blog', 'r-vattention']
    },

    {
      id: 'u-prefix-caching', tier: 2,
      title: 'Prefix caching: the highest-leverage feature you already have',
      summary: 'Hash-keyed block reuse turns repeated system prompts and multi-turn chat into near-zero prefill.',
      why: 'Real traffic is enormously repetitive. Every request carries the same system prompt. Every turn of a conversation re-sends the entire history. Every few-shot template repeats the same exemplars. Agent loops re-send a growing but mostly identical context. Without caching, you re-prefill all of that on every single request.\n\nAutomatic prefix caching hashes each KV block\'s content plus its prefix and looks it up before computing. On a chat workload with a 2000-token system prompt and 10-turn conversations, hit rates of 70-90% are ordinary, and TTFT drops by roughly the hit rate. It is usually the single largest available win, it costs nothing but memory pressure, and a surprising number of deployments leave it off or defeat it accidentally.',
      core: [
        'The cache key is a hash chain: block `i`\'s hash depends on its token ids *and* the hash of block `i-1`. This makes a hit mean "identical prefix", not merely "identical block".',
        'Only full blocks are cacheable. With block size 16, a shared prefix of 1000 tokens yields 62 reusable blocks and 8 recomputed tokens.',
        'Cached blocks live in the same pool as active ones and are evicted (usually LRU) under pressure. So prefix cache hit rate falls as concurrency rises — the two compete for the same memory.',
        'Anything that varies at the *front* of the prompt destroys all downstream reuse. Timestamps, request ids, or per-user data placed before the system prompt are the classic own-goal.',
        'Cache-aware routing multiplies the benefit: send requests with the same prefix to the same replica, or hit rate collapses to `1/n_replicas`.',
        'Hit rate is a first-class metric. Export it, alert on it, and treat a drop as an incident — it usually means someone reordered the prompt template.',
        'Security: prefix cache sharing across tenants can, in principle, leak timing information about other tenants\' prompts. Most engines scope caches per-tenant or accept the risk knowingly.'
      ],
      math: [{
        title: 'What prefix caching buys',
        body: `TTFT_effective ~= TTFT_full * (1 - hit_rate) + lookup_overhead

Multi-turn chat, turn k:
    context_k    = system + sum_{i<k}(user_i + assistant_i) + user_k
    cacheable    = everything except user_k (and partial last block)
    hit_rate_k  ~= 1 - len(user_k)/len(context_k)

Turn 8 with 4000 tokens of history and a 60-token new message:
    hit_rate ~= 98.5%  ->  prefill cost falls ~65x

Prefill FLOPs saved per hour =
    2 * P * cached_tokens_per_request * requests_per_hour`,
        note: 'Convert that to dollars with the cost calculator. On a chat product it is routinely the difference between profitable and not.'
      }],
      pitfalls: [
        'Injecting a timestamp, request id, or random nonce at the top of the system prompt. This alone can take hit rate from 90% to 0%.',
        'Round-robin load balancing across replicas without cache awareness, which divides hit rate by the replica count.',
        'Assuming a hit avoids all work. The kernel still reads the cached KV during decode; you save prefill compute, not decode bandwidth.',
        'Letting high concurrency evict the shared prefix repeatedly (thrashing). Some engines let you pin or prioritize; otherwise raise memory or shard by prefix.',
        'Measuring hit rate in tokens vs in blocks vs in requests and comparing them. Pick one definition and state it.'
      ],
      build: [
        'Build a synthetic multi-turn workload with a fixed 2000-token system prompt and 10 turns. Measure TTFT and reported prefix-cache hit rate with caching on and off.',
        'Deliberately break it by prepending `str(uuid4())` to the system prompt. Record the new hit rate and TTFT. Keep the numbers; this is the most persuasive slide you will ever show a product team.',
        'If you run more than one replica, implement or configure consistent-hash routing on the prompt prefix and measure hit rate before and after.'
      ],
      checks: [
        'Why does block `i`\'s hash need to include block `i-1`\'s hash?',
        'A 1000-token shared prefix with block size 16: how many blocks are reused and how many tokens recomputed?',
        'Under what conditions does raising concurrency *reduce* total throughput because of prefix caching?'
      ],
      res: ['r-modular-prefix', 'r-vllm-prefix-docs', 'r-sglang-blog', 'r-cacheblend', 'r-modular-routing']
    },

    {
      id: 'u-radix', tier: 3,
      title: 'RadixAttention and the shape of the cache',
      summary: 'A radix tree over token sequences: the alternative to hash-keyed blocks, and what it wins.',
      why: 'vLLM stores KV in fixed blocks keyed by a hash chain. SGLang stores it in a radix tree whose edges are token sequences and whose nodes hold KV. The two solve the same problem — reuse shared prefixes — with different data structures, and the differences show up exactly where you would expect: branching workloads, partial-block sharing, and eviction granularity.\n\nStudying both is the fastest way to understand the design space rather than one product. The radix tree naturally represents tree-structured generation (self-consistency, tree-of-thought, speculative branches, agent forks), supports LRU eviction at node granularity, and enables cache-aware scheduling that reorders the batch to maximize hits.',
      core: [
        'A radix tree keyed by token ids: each node holds the KV for its edge label; a shared prefix is a shared path from the root.',
        'Matching is longest-prefix, so partial matches are naturally handled without block alignment constraints.',
        'Eviction is LRU over leaves, with reference counting so nodes in use by running requests are not evicted.',
        'Cache-aware scheduling reorders the waiting queue to group requests that share a prefix, which raises hit rate beyond what arrival order would give.',
        'LMSYS reported up to 5x throughput on prefix-heavy workloads with no measurable overhead when there are no hits — the second half of that claim is the important one.',
        'vLLM\'s hash-chain approach later achieved much of the same benefit with simpler machinery. The interesting question is not which is faster today but which invariants each gives up.',
        'Both approaches interact with paging: SGLang also pages memory; the radix tree is the index, not the allocator.'
      ],
      tables: [{
        title: 'Two designs for the same problem',
        cols: ['Aspect', 'Hash-chained blocks (vLLM)', 'Radix tree (SGLang)'],
        rows: [
          ['Sharing granularity', 'Whole blocks only (e.g. 16 tokens)', 'Arbitrary token boundaries'],
          ['Lookup', 'Hash map, O(1) per block', 'Tree walk, O(prefix length)'],
          ['Branching workloads', 'Works, via ref counts', 'Native representation'],
          ['Eviction', 'LRU over blocks', 'LRU over leaves, ref-counted'],
          ['Scheduling integration', 'Independent of scheduler', 'Cache-aware batch reordering'],
          ['Implementation cost', 'Low', 'Higher; tree maintenance on every step']
        ]
      }],
      pitfalls: [
        'Assuming a 5x headline transfers to your workload. It applies to prefix-heavy traffic; on unique-prompt traffic both designs give roughly nothing.',
        'Comparing engines on a benchmark whose prompts are randomly generated. That deliberately defeats the feature you are trying to measure.',
        'Ignoring that tree maintenance is CPU work on the critical path, which matters in the overhead-bound regime.'
      ],
      build: [
        'Run the identical shared-prefix workload against vLLM and SGLang on the same hardware and model. Report TTFT p50/p99 and hit rate for both. Then run a zero-sharing workload and confirm neither engine loses.',
        'Sketch (on paper) the radix tree state after five requests that share a system prompt and branch into three different follow-ups. Mark which nodes are ref-counted.'
      ],
      checks: [
        'Which workload makes a radix tree strictly better than block hashing, and why?',
        'Why does cache-aware scheduling raise hit rate above what the cache alone provides?',
        'What is the cost of the radix tree when hit rate is zero, and why does that matter?'
      ],
      res: ['r-sglang-blog', 'r-sglang-paper', 'r-sglang-repo', 'r-sglang-v04', 'r-preble']
    },

    {
      id: 'u-eviction', tier: 3,
      title: 'KV eviction and sparse attention for long context',
      summary: 'Attention sinks, heavy hitters, and the policies that make 128k context affordable.',
      why: 'At long context the KV cache dominates memory and decode bandwidth, and it grows without bound. The observation that rescues this is that attention is extremely concentrated: a small set of tokens receives most of the attention mass, and the first few tokens of a sequence act as "sinks" that absorb probability mass regardless of their content. That means most of the cache can be dropped or compressed with small quality loss — if you drop the right parts.\n\nThe literature here is large and the practical guidance is narrow: keep the sinks, keep a recent window, and keep the heavy hitters; select rather than evict when you can afford to store the cache elsewhere. This unit is about knowing which policy suits which workload, and about measuring the damage honestly rather than trusting a perplexity number.',
      core: [
        'StreamingLLM: keeping the first few tokens (attention sinks) plus a rolling window enables stable generation over millions of tokens. Removing the sinks alone collapses quality — the finding is about softmax needing somewhere to dump mass.',
        'H2O: track accumulated attention scores and evict the low-scoring "non-heavy-hitters". Greedy and effective, but the eviction is irreversible.',
        'SnapKV / PyramidKV: compress the *prompt* KV at prefill time using observed attention patterns, with a per-layer budget that shrinks with depth. Good for long-prompt/short-output workloads such as RAG.',
        'Quest / MInference and other selective-attention methods keep the full cache but *select* a subset of pages per query, converting a bandwidth problem into an index problem. Nothing is lost permanently, which makes them safer for retrieval tasks.',
        'DuoAttention and similar split heads into "retrieval heads" (need full cache) and "streaming heads" (need only sink + window), giving a principled per-head budget.',
        'Every eviction policy has an adversarial workload: needle-in-a-haystack retrieval, long-range code edits, and multi-hop reasoning are where they break. Evaluate on those, not on perplexity.',
        'Eviction interacts badly with prefix caching: an evicted-then-restored prefix is no longer a cache hit. Decide which one owns the memory.'
      ],
      math: [{
        title: 'Budgeted KV per request',
        body: `Full cache       : kv_per_token * S
StreamingLLM     : kv_per_token * (n_sink + w)            # constant in S
H2O budget b     : kv_per_token * b                       # constant
Pyramid budget   : kv_per_token * sum_l b_l, b_l decreasing with depth
Selective (Quest): full cache stored, but only k pages read per step
                   -> memory unchanged, bandwidth ~ k/S of full

Decode bandwidth per step, batch B:
    weights + B * kv_per_token * effective_tokens_read`,
        note: 'Note the difference in what each policy optimizes. Eviction reduces memory *and* bandwidth. Selection reduces bandwidth only. If you are capacity limited, selection does not help you.'
      }],
      pitfalls: [
        'Evaluating eviction with perplexity. Perplexity is dominated by local context and will barely move while retrieval accuracy collapses.',
        'Applying a uniform per-layer budget. Attention concentration varies strongly by layer and by head.',
        'Forgetting the sinks. Almost every "sliding window destroyed my model" report is a missing sink.',
        'Enabling eviction on a workload whose outputs are then used for evaluation, and attributing the quality drop to the base model.'
      ],
      build: [
        'Run a needle-in-a-haystack style retrieval eval at 32k and 128k context with (a) full cache, (b) FP8 KV, (c) one eviction policy at 50% budget. Produce a single table of accuracy, peak KV memory and ITL.',
        'Plot decode ITL against context length for full versus budgeted cache and identify the crossover where the policy starts to pay.'
      ],
      checks: [
        'Why do attention sinks exist at all? Give the softmax argument.',
        'When is selective attention preferable to eviction, and when is it useless?',
        'You must serve 200k-token contexts on fixed hardware. Order your options by risk to quality.'
      ],
      res: ['r-streamingllm', 'r-h2o', 'r-snapkv', 'r-pyramidkv', 'r-quest', 'r-duoattention', 'r-minference']
    },

    {
      id: 'u-kv-quant', tier: 3,
      title: 'KV cache quantization',
      summary: 'FP8 and INT4 KV: usually the cheapest capacity you will ever buy, with a real but small quality bill.',
      why: 'Halving KV precision doubles your concurrency at long context and halves the decode bandwidth spent on the cache. On a long-context workload where KV traffic exceeds weight traffic, that is close to halving decode latency. Very few optimizations have that ratio of benefit to effort — it is usually a single flag.\n\nThe catch is that keys and values behave differently. Keys have strong per-channel outliers (largely from RoPE) and want per-channel scaling; values are better behaved and tolerate per-token scaling. Naive uniform INT4 on both collapses quality; the papers that work (KIVI, KVQuant) get there by treating them asymmetrically and keeping a small outlier set in higher precision.',
      core: [
        'FP8 (E4M3) KV is close to free in quality on most models and is the default recommendation. INT4 KV needs a careful scheme.',
        'Keys: quantize per-channel. Values: per-token. This asymmetry is the central empirical result of the KV quantization literature.',
        'Keep a small residual window of recent tokens in full precision; recent KV disproportionately affects the next token.',
        'Quantized KV changes the attention kernel: dequantize in registers, accumulate in FP32. If your backend lacks a fused kernel, you pay in compute what you saved in bandwidth.',
        'The win is proportional to how KV-dominated your workload is. At 512-token contexts, KV quantization does almost nothing; at 128k it can dominate everything else.',
        'Measure quality on long-context retrieval, not on short-form benchmarks. The damage from KV quantization concentrates exactly where the cache matters.',
        'KV quantization and weight quantization are independent knobs with independent failure modes. Change one at a time.'
      ],
      math: [{
        title: 'Bandwidth and capacity effect',
        body: `Decode bytes/step = P*w_bytes + B * S * kv_bytes_per_token

Llama-3-70B, B=32, S=32768, TP=8:
  weights (BF16)        = 140 GB
  KV FP16 (320 KiB/tok) = 32 * 32768 * 327680 = 343 GB   <-- dominates
  KV FP8                = 172 GB
  KV INT4               = 86 GB

Step time (8x H100, 80% eff, 26.8 TB/s):
  FP16 KV : (140+343)/26800 = 18.0 ms
  FP8  KV : (140+172)/26800 = 11.6 ms   -> 1.55x faster decode
  INT4 KV : (140+86)/26800  =  8.4 ms   -> 2.14x faster decode`,
        note: 'At long context the KV cache, not the weights, is the thing you are reading. This is the calculation that surprises people who have only ever tuned weight quantization.'
      }],
      pitfalls: [
        'Enabling FP8 KV and comparing throughput without checking whether the engine fell back to a slower attention backend that supports it.',
        'Quantizing keys per-token. RoPE creates channel-structured outliers; per-token scaling on keys is the standard way to destroy accuracy.',
        'Assuming FP8 KV halves memory exactly. Scales and zero points cost a few percent, and some implementations keep a full-precision window.',
        'Testing on short prompts, shipping, and discovering the regression only on 100k-token support tickets.'
      ],
      build: [
        'Benchmark FP16 vs FP8 KV at 4k, 32k and 128k context: report cacheable tokens, ITL, and retrieval accuracy for each.',
        'Compute, for your workload, the context length at which KV bytes exceed weight bytes in the decode step. That is the point where KV quantization becomes your top priority.'
      ],
      checks: [
        'Why do keys and values want different quantization granularity?',
        'At what context length does KV traffic exceed weight traffic for a 70B model at batch 32, BF16 weights, GQA-8?',
        'Why can KV quantization sometimes *reduce* throughput?'
      ],
      res: ['r-kivi', 'r-kvquant', 'r-vllm-quant-docs', 'r-modular-kv-offload', 'r-qserve']
    },

    {
      id: 'u-offload', tier: 3,
      title: 'Offload and the KV memory hierarchy',
      summary: 'GPU to CPU to NVMe to network: when moving KV beats recomputing it.',
      why: 'Once you accept that KV is the scarce resource, the natural next step is a memory hierarchy. Cold prefixes can live in CPU DRAM, on NVMe, or in a cluster-wide KV store, and be pulled back when a matching request arrives. Mooncake built an entire serving architecture around this idea; LMCache productized it; every disaggregated design depends on it.\n\nThe decision is always the same comparison: is transferring `n` tokens of KV cheaper than recomputing it? Recomputation costs prefill FLOPs; transfer costs bandwidth over PCIe, NVLink or the network. The crossover is a concrete number you can compute for your hardware, and it moves with context length, model size and link speed.',
      core: [
        'Transfer cost = `kv_bytes_per_token × n / link_bandwidth`. Recompute cost = `2 × P × n / effective_prefill_FLOPs`. Set them equal and solve for the link speed at which transfer wins.',
        'PCIe Gen5 x16 is ~64 GB/s unidirectional. NVLink within a node is hundreds of GB/s. A 400 Gb/s NIC is ~50 GB/s. These three regimes give very different answers.',
        'GQA and MLA make transfer far more attractive by shrinking bytes per token; MHA models are much more likely to favour recompute.',
        'CPU offload also serves preemption: swap a preempted sequence out rather than dropping and recomputing it. Which is better depends on how much KV the sequence holds.',
        'Cluster-wide KV stores turn prefix caching into a shared service, so a cache hit on any replica benefits all. This is what makes cache-aware routing worth building.',
        'Every offload path adds a tail-latency risk. Measure p99 with the offload path active and cold, not just warm.'
      ],
      math: [{
        title: 'Transfer versus recompute',
        body: `t_transfer(n) = n * kv_bytes_per_token / BW_link
t_recompute(n) = 2 * P * n / (peak_FLOPs * MFU)

Break-even link bandwidth:
    BW* = kv_bytes_per_token * peak_FLOPs * MFU / (2 * P)

Llama-3-70B, GQA-8: kv/token = 327,680 B, P = 70e9
H100 prefill at 45% MFU: 989e12 * 0.45 = 445 TFLOP/s

BW* = 327680 * 445e12 / (2 * 70e9) = 1.04 TB/s

-> Transfer only wins over NVLink-class links (or better).
   Over PCIe (64 GB/s) recompute is ~16x cheaper. Over a 400G NIC, worse.`,
          note: 'Now redo this for an MHA model (kv/token 4x larger) and for an MLA model (kv/token 5x smaller) and note that the architectural choice flips the answer. This is why Mooncake-style designs pair naturally with MLA models.'
      }],
      pitfalls: [
        'Quoting bidirectional link bandwidth in a one-way transfer calculation.',
        'Ignoring that the transfer competes with the very same link your tensor-parallel all-reduces use.',
        'Offloading synchronously on the critical path instead of prefetching against the scheduler\'s lookahead.',
        'Building a distributed KV store before measuring the single-node prefix cache hit rate. Usually the simple thing was enough.'
      ],
      build: [
        'Compute the break-even bandwidth for your model and hardware. State clearly whether offload can ever pay for you.',
        'If you have LMCache or an equivalent available, measure TTFT for a cold prefix, a CPU-resident prefix, and a GPU-resident prefix. Three numbers, one table.'
      ],
      checks: [
        'Derive the break-even bandwidth formula from scratch.',
        'Why does MLA make KV transfer more attractive than GQA, and GQA more attractive than MHA?',
        'When is swap-based preemption better than recompute-based preemption?'
      ],
      res: ['r-mooncake', 'r-lmcache', 'r-modular-kv-offload', 'r-flexgen', 'r-nixl']
    }
  ]
},

/* ==================================================================== 03 */
{
  id: 'm03', num: '03',
  title: 'The scheduler: batching, admission, preemption',
  tagline: 'Continuous batching, chunked prefill, and the code that decides what runs each step.',
  focus: 'The scheduler is the part of an inference engine that determines almost all of its observable behaviour: throughput, TTFT, ITL jitter, fairness, and failure mode under overload. It is also the part with the least literature and the most impact. This module works from the Orca insight (schedule at iteration granularity, not request granularity) through chunked prefill and stall-free scheduling, into the actual scheduler loop of a modern engine.',
  outcome: 'A written trace of one scheduler step in vLLM V1 with the token budget accounted for, plus measured TTFT/ITL curves showing the chunked-prefill trade on your own service.',
  units: [
    {
      id: 'u-continuous-batching', tier: 2,
      title: 'From static to continuous batching',
      summary: 'Iteration-level scheduling: the single idea that made LLM serving economical.',
      why: 'Static batching waits for a batch to form, runs it to completion, and returns. Because generation lengths vary by an order of magnitude within any real batch, the GPU spends most of its time computing padding for sequences that already finished. Orca\'s contribution was to schedule at the granularity of a single decoding iteration: after every step, finished sequences leave and waiting ones join.\n\nThe measured effect is large — Anyscale reported over 20x throughput against naive static batching — and the reason is exactly the arithmetic intensity argument from Module 00. Continuous batching is not a queueing nicety; it is the mechanism by which decode arithmetic intensity is kept high. Everything else in the scheduler is a refinement of this idea plus a resource constraint.',
      core: [
        'Static batching: batch size fixed for the lifetime of the batch. Utilization collapses as sequences finish at different times, and tail latency is set by the longest generation.',
        'Dynamic batching (from classic serving) forms batches with a timeout. It helps for fixed-cost models and does almost nothing for autoregressive generation.',
        'Continuous / in-flight batching: the batch is re-formed every iteration. New requests are admitted as soon as memory allows; finished ones free their blocks immediately.',
        'Orca also introduced selective batching: operations that can be batched across sequences with different lengths (the GEMMs) are, while attention is handled per-sequence.',
        'The binding constraint on batch size is KV memory, not compute. This is why paging and batching arrived together — neither is very useful without the other.',
        'A batch is heterogeneous: some sequences are prefilling, some decoding. How you mix them is the subject of the next unit and is where TTFT and ITL trade against each other.',
        'Fairness is not free. Pure throughput-maximizing admission starves long requests; most engines apply FCFS with a priority hook and let you override it.'
      ],
      math: [{
        title: 'Why static batching wastes so much',
        body: `Batch of B sequences with output lengths L_1..L_B.

Static: total steps = max(L_i), useful token-steps = sum(L_i)
    utilization = sum(L_i) / (B * max(L_i))

Exponentially distributed lengths, mean m, B=32:
    E[max] ~= m * (ln B + gamma) ~= m * 3.9
    utilization ~= 1/3.9 ~= 26%

Continuous: utilization ~= 1 (limited by memory and arrivals)`,
        note: 'The heavier the tail of your output-length distribution, the larger the win. Chat traffic, where a few requests generate 20x the median, is close to the worst case for static batching and therefore the best case for continuous.'
      }],
      pitfalls: [
        'Setting `max_num_seqs` very high and assuming throughput rises. Past the KV limit you get preemption, not throughput.',
        'Assuming continuous batching fixes TTFT. It fixes utilization; TTFT under load is governed by admission and prefill scheduling.',
        'Benchmarking with uniform output lengths, which erases the entire advantage you are trying to measure.',
        'Ignoring that a request holds its KV for its whole lifetime, so long generations occupy memory that short ones could have cycled through.'
      ],
      build: [
        'Generate a workload with output lengths drawn from a heavy-tailed distribution. Compare throughput with `max_num_seqs=1` (effectively serial), a small value, and the engine default. Plot.',
        'Read Orca\'s selective batching section and identify, in your engine\'s code, where batched and per-sequence operations diverge.'
      ],
      checks: [
        'Why does variance in output length hurt static batching more than mean length does?',
        'What resource, not compute, actually limits the batch size in a modern engine?',
        'Give a workload for which continuous batching provides almost no benefit.'
      ],
      res: ['r-orca', 'r-anyscale-batching', 'r-modular-batching', 'r-vllm-paper', 'r-deepspeed-fastgen']
    },

    {
      id: 'u-chunked-prefill', tier: 2,
      title: 'Chunked prefill and stall-free scheduling',
      summary: 'The knob that trades TTFT against ITL, and why it became the default everywhere.',
      why: 'Mixing prefill and decode in one engine creates a direct conflict. A 16k-token prefill takes hundreds of milliseconds; if it runs as one unit, every decoding request in flight stalls for that whole time and every user sees a stutter. Prioritize decode instead and prefills starve, so TTFT explodes.\n\nSarathi-Serve\'s answer is to split a long prefill into token-budgeted chunks and pack each chunk into a step alongside decodes. Each step now has a roughly constant token budget, so step time is roughly constant, so ITL is smooth. TTFT for the long prefill rises slightly because it is spread over more steps. That trade — a little TTFT for a lot of ITL stability — is why chunked prefill is the default in vLLM and SGLang, and understanding the knob is how you tune it back when your workload wants the opposite.',
      core: [
        'A step has a token budget (`max_num_batched_tokens`). Decodes cost 1 token each; a prefill chunk consumes the remainder.',
        'Larger budget: better prefill throughput and TTFT, worse ITL (longer, lumpier steps). Smaller budget: smooth ITL, worse TTFT and lower prefill efficiency.',
        'Prefill chunks below a few hundred tokens stop being compute-efficient — you are re-reading weights for too little work. There is a floor, and it is hardware dependent.',
        'Piggybacking decodes onto a prefill-heavy step is nearly free, because the step is already compute bound. This is the "stall-free" part of stall-free batching.',
        'The interaction with prefix caching matters: a cache hit shortens the prefill, changing how many chunks it needs and therefore its scheduling behaviour.',
        'Under disaggregation (Module 07) this tension disappears differently: prefill and decode run on separate hardware, so neither needs to be chunked to protect the other.',
        'Attention for a chunk must attend to all previously computed KV of the same request, so chunked prefill needs a kernel that handles "query length > 1, key length much longer" efficiently.'
      ],
      math: [{
        title: 'The token budget',
        body: `tokens_in_step = n_decode_requests * 1 + prefill_chunk_tokens
                 <= max_num_batched_tokens

Step time ~ linear in tokens_in_step once compute bound.

ITL_perceived ~= step_time (a decode advances one token per step)

TTFT(prompt T) ~= queue_wait
                 + ceil(T / chunk) * step_time
                 (chunk = budget - n_decode)

Trade:
  budget 512   -> steps short, ITL smooth, long prompts take many steps
  budget 8192  -> few steps, fast TTFT, ITL spikes when a chunk lands`,
        note: 'Tune `max_num_batched_tokens` by measuring, not by intuition. Sweep it over at least 512, 1024, 2048, 4096, 8192 with your real prompt-length distribution and plot TTFT p99 and ITL p99 on the same axes.'
      }],
      pitfalls: [
        'Tuning the token budget on a workload with uniform short prompts and then deploying on long-prompt traffic.',
        'Setting the budget below the decode batch size, which starves prefill entirely.',
        'Assuming chunked prefill always helps. For pure batch/offline generation with no latency SLO, large unchunked prefills are more efficient.',
        'Forgetting that with chunked prefill enabled, "prefill throughput" measured in isolation is no longer what the service does.'
      ],
      build: [
        'Sweep `max_num_batched_tokens` across five values on a mixed workload (long prompts + concurrent decodes). Plot TTFT p50/p99 and ITL p50/p99. Identify the setting that maximizes goodput under your SLO.',
        'Construct the pathological case: one 32k prompt arriving while 64 requests decode. Measure the ITL spike with chunking disabled and enabled.'
      ],
      checks: [
        'Explain precisely why chunked prefill improves ITL but can worsen TTFT.',
        'Why is there a lower bound on useful chunk size?',
        'Under what deployment does chunked prefill become unnecessary?'
      ],
      res: ['r-sarathi', 'r-vllm-optimization-docs', 'r-deepspeed-fastgen', 'r-sglang-blog', 'r-nanoflow']
    },

    {
      id: 'u-admission', tier: 3,
      title: 'Admission, preemption, and behaviour under overload',
      summary: 'What the scheduler does when it runs out of memory, and why p99 lives here.',
      why: 'Every serving system is fine until it is saturated. What happens next is a design decision most people never make deliberately: the engine admits a request it cannot finish, runs out of KV blocks, and must evict something. Whether it recomputes or swaps, whether it evicts the newest or the largest, and whether it applies backpressure to the queue determines whether your p99 degrades gracefully or falls off a cliff.\n\nThis is also where most production incidents come from. Throughput graphs look fine; p99 TTFT goes to thirty seconds because requests are being admitted, partially processed, preempted, and restarted. Understanding the preemption path and instrumenting it is the difference between debugging this in minutes and in days.',
      core: [
        'Admission control decides whether a waiting request can start. Engines typically admit if projected KV need fits within a watermark; the projection uses `max_tokens`, which users routinely overstate.',
        'Preemption strategies: recompute (drop KV, re-prefill later) or swap (copy KV to CPU and back). Recompute costs prefill FLOPs and is preferred for short contexts; swap costs PCIe bandwidth and is preferred for long ones.',
        'A preempted request that is re-admitted and preempted again is livelock. Good schedulers apply priority aging or admit-and-hold to prevent it.',
        'The right overload behaviour is usually to *reject or queue* rather than admit. A 503 with a retry hint is a better user experience than a 40-second TTFT.',
        'Key metrics to export: `num_requests_waiting`, `num_requests_running`, KV cache utilization, preemption count, and the queue time component of TTFT. Without these, overload is invisible.',
        'Priority and fairness: FCFS is the default and is fine until one tenant sends 10k-token prompts. Then you need per-tenant budgets or weighted fair queueing.',
        'Autoscaling should trigger on queue depth or KV utilization, not on GPU utilization — GPU utilization is near 100% by design and carries no information.'
      ],
      math: [{
        title: 'Overload dynamics',
        body: `Stable iff  arrival_rate < service_rate.

Queue time under M/M/1-ish approximation:
    W_q = rho / (mu * (1 - rho))      rho = lambda/mu

rho = 0.8 -> W_q = 4 / mu
rho = 0.95 -> W_q = 19 / mu
rho = 0.99 -> W_q = 99 / mu

TTFT = W_q + prefill_time. This is why TTFT p99 explodes
long before throughput visibly saturates: the knee is at
rho ~ 0.8, not 1.0.

With preemption, effective service rate FALLS as load rises
(work is repeated), so the system can become unstable at
rho < 1. That is the cliff.`,
        note: 'The last point is the important one. Preemption-heavy overload is a positive feedback loop: more load causes more preemption causes less effective throughput causes more load in the queue.'
      }],
      pitfalls: [
        'Autoscaling on GPU utilization. It will read ~95% at idle-ish load and give you no signal.',
        'Allowing unbounded `max_tokens` from clients, which makes admission projections meaningless.',
        'Setting queue length to infinity. An unbounded queue converts a capacity problem into a latency problem and hides it from your monitoring.',
        'Load testing only at the operating point. The interesting behaviour is at 1.2x capacity, and you need to know it before your users find it.'
      ],
      build: [
        'Drive your service to 1.5x its measured capacity for five minutes. Record TTFT percentiles, preemption count, and queue depth over time. Write down the failure mode in one paragraph.',
        'Implement or configure a queue-depth-based rejection policy and repeat. Compare p99 and error rate.'
      ],
      checks: [
        'Why can a system with preemption become unstable below 100% nominal utilization?',
        'When is swap-based preemption better than recompute, quantitatively?',
        'What signal should drive autoscaling for an LLM service, and why not GPU utilization?'
      ],
      res: ['r-vllm-repo', 'r-llumnix', 'r-vllm-optimization-docs', 'r-fastserve', 'r-modular-routing']
    },

    {
      id: 'u-vllm-v1', tier: 3,
      title: 'Reading the vLLM V1 engine',
      summary: 'One step of the scheduler loop, traced through actual source files.',
      why: 'You can read every paper in this curriculum and still not know how a serving engine works, because the papers describe mechanisms and the engine is mostly the *composition* of mechanisms plus a great deal of pragmatic engineering. The V1 rewrite is a good object of study: it separates the API server process from the engine core, moves scheduling to a flat token-budget model, makes prefix caching always-on, and reduces per-step CPU work so the GPU is not waiting on Python.\n\nThe goal here is not to memorize file paths, which change. It is to be able to answer, for any engine, the same five questions: what forms a step, who owns memory, what happens on overload, where the tokenizer runs, and where the CPU time goes.',
      core: [
        'Process split: an API/front-end process handles HTTP, tokenization and detokenization; the engine core process runs the scheduler and model. They communicate over IPC so Python overhead in one does not stall the other.',
        'The scheduler builds a step from a token budget rather than distinguishing "prefill batch" and "decode batch". Prefills and decodes are just token allocations.',
        'The KV cache manager owns block allocation, hashing, and prefix cache lookup; the scheduler asks it whether a request can be admitted.',
        'Model execution is behind a worker abstraction, with attention backends selected at runtime by GPU, dtype, head dim and paging requirements.',
        'CUDA graphs are captured for a configured set of batch sizes; requests are padded up to the nearest captured size.',
        'Piecewise `torch.compile` compiles the model in fragments so graph breaks around attention do not defeat the whole thing.',
        'Where CPU time goes: tokenization, detokenization, sampling parameter handling, scheduler bookkeeping, and output serialization. Each has been separately attacked in the V1 design.'
      ],
      code: [{
        title: 'A reading plan, with the questions you must answer',
        body: `1. Entry:      vllm/entrypoints/openai/api_server.py
   -> Where does a request become an EngineCoreRequest?

2. Scheduler:  vllm/v1/core/sched/scheduler.py
   -> Trace schedule(): how is the token budget spent?
   -> Where are prefill chunks sized?
   -> Where does preemption happen and which policy?

3. Memory:     vllm/v1/core/kv_cache_manager.py, block_pool.py
   -> What is hashed for prefix caching?
   -> When is a block returned to the free pool vs kept cached?

4. Execution:  vllm/v1/worker/gpu_model_runner.py
   -> How is the batch laid out for the attention kernel?
   -> Where are CUDA graphs replayed, and what is padded?

5. Output:     vllm/v1/engine/output_processor.py
   -> Where does detokenization run, and on which process?`,
        note: 'Write the answers down. If you can answer these five for vLLM, you can answer them for SGLang or TensorRT-LLM in an afternoon, because the questions are the invariant part.'
      }],
      pitfalls: [
        'Reading the paper and assuming the code matches it. The paper describes V0-era design; V1 differs in significant ways.',
        'Benchmarking a build with debug logging or assertions enabled.',
        'Assuming flags mean the same thing across versions. Check the version you are running against the docs for that version.',
        'Trying to understand the engine by reading the model definitions. The interesting logic is in the scheduler and memory manager.'
      ],
      build: [
        'Trace one scheduler step end-to-end with logging enabled and produce a written narrative: which requests were selected, how the token budget was divided, how many blocks were allocated, and what was preempted.',
        'Change one scheduler-relevant flag (`max_num_batched_tokens`) and re-trace. Explain the difference in the log, not in the throughput number.'
      ],
      checks: [
        'What forms a "step" in V1, and why is that framing better than prefill/decode batches?',
        'Which process does detokenization run in, and why does that choice matter?',
        'How does the engine decide which attention backend to use?'
      ],
      res: ['r-vllm-repo', 'r-vllm-v1-blog', 'r-vllm-docs', 'r-vllm-paper', 'r-gpu-mode-l35']
    },

    {
      id: 'u-sampling', tier: 2,
      title: 'Sampling, logits processing, and structured output',
      summary: 'The tail end of every step, and a surprisingly common source of latency and bugs.',
      why: 'Sampling looks trivial and is not. The logits tensor is `[batch, vocab]` — for a 128k vocabulary and batch 256 that is 32M floats, and every penalty, bias, mask and sort over it is real bandwidth. Structured output adds a per-request grammar mask that must be computed on CPU and applied on GPU every single step, and a naive implementation serializes the entire batch behind it.\n\nThere is also a correctness dimension that matters for evaluation: temperature, top-p, top-k, min-p, repetition penalties, and seeds must be identical when comparing two engines, or you are measuring sampler differences and calling them engine differences.',
      core: [
        'Greedy, temperature, top-k, top-p (nucleus), min-p, and typical sampling are all masks or transforms on the logit distribution. min-p scales the threshold by the top probability and behaves better at high temperature.',
        'Penalties (presence, frequency, repetition) require per-request token histories and are gather-heavy; engines batch them, but they are not free.',
        'Structured/constrained decoding builds a finite-state machine (Outlines) or a pushdown automaton over a grammar (XGrammar) and masks disallowed tokens each step. The mask computation can be overlapped with the forward pass; if it is not, it is on the critical path.',
        'Compiling a JSON schema to an FSM is expensive but cacheable across requests with the same schema. Cache it or you pay per request.',
        'Speculative decoding interacts with sampling: rejection sampling must preserve the target distribution exactly, which constrains what sampler transformations can be applied and when.',
        'For reproducible benchmarks, pin the seed, the sampler parameters, and the tokenizer. Report all three.',
        'Detokenization is incremental and surprisingly fiddly with multi-byte characters and special tokens; doing it on the engine\'s hot path costs throughput.'
      ],
      math: [{
        title: 'What sampling costs',
        body: `logits tensor = batch * vocab * 4 bytes (FP32)
  batch 256, vocab 128k -> 131 MB

Reading and writing it a few times per step:
  ~0.4 GB of traffic at 3.35 TB/s = 0.12 ms

A full sort for top-k over 128k vocab is far worse than a
partial selection; engines use radix/top-k kernels.

Grammar mask: batch * vocab bits = 256 * 128k / 8 = 4 MB per step,
computed on CPU. At 20 ms/step this must complete in <20 ms
for the whole batch, or it becomes the bottleneck.`,
        note: 'On small models with short steps, the sampler and the grammar mask can genuinely exceed the model forward pass. Profile it before assuming otherwise.'
      }],
      pitfalls: [
        'Comparing two engines with different default `top_p` or `repetition_penalty`. This is the single most common invalid benchmark.',
        'Leaving structured output on for requests that do not need it.',
        'Recompiling the grammar per request instead of caching by schema hash.',
        'Measuring token counts with a different tokenizer than the model uses, which silently changes your throughput number by 5-20%.'
      ],
      build: [
        'Measure throughput with greedy decoding, with top-p sampling plus penalties, and with a JSON schema constraint. Three numbers on the same workload.',
        'Time grammar compilation for a moderately complex schema and confirm your stack caches it.'
      ],
      checks: [
        'Why does min-p behave better than top-p at high temperature?',
        'How does constrained decoding interact with speculative decoding\'s rejection sampling?',
        'Where in a step does the grammar mask have to be applied, and what can be overlapped?'
      ],
      res: ['r-outlines', 'r-xgrammar', 'r-nucleus-sampling', 'r-vllm-docs', 'r-sglang-paper']
    }
  ]
},

/* ==================================================================== 04 */
{
  id: 'm04', num: '04',
  title: 'Measurement: build the lens before the optimizations',
  tagline: 'Benchmarking methodology, instrumentation, load generation, and kernel profiling.',
  focus: 'Nothing after this module is verifiable without it. A published benchmark is mostly a credible measurement harness with a model attached, and almost every inference claim you will read on the internet fails at least one of the criteria in this module. Build the lens now, then everything downstream becomes a controlled experiment instead of an anecdote.',
  outcome: 'A reproducible concurrency sweep with p50/p95/p99 for TTFT and ITL, cross-checked by two independent tools, backed by a live dashboard and a pinned-version writeup that a skeptic could rerun.',
  units: [
    {
      id: 'u-bench-method', tier: 2,
      title: 'What makes an inference benchmark defensible',
      summary: 'The variables you must pin, the ones you must sweep, and the ones that invalidate everything.',
      why: 'An LLM inference benchmark has more confounds than almost any other systems benchmark: input length, output length, their distributions, concurrency, sampling parameters, tokenizer, quantization, KV dtype, engine version, attention backend, GPU clocks, and whether prefix caching was hitting. Change any one silently and the number moves by 2x.\n\nThe fix is a discipline, not a tool. Pin everything you are not studying, sweep the load, report distributions rather than means, and publish the exact command. If a benchmark does not let a reader reproduce it, it is a marketing artifact.',
      core: [
        'Fix input and output token lengths explicitly, and say whether they are fixed or sampled from a distribution. Output length determines KV size and therefore the entire result.',
        'Sweep concurrency from 1 (synchronous baseline) to saturation. One point is not a benchmark.',
        'Report p50, p95 and p99 for TTFT and ITL, plus output-token throughput. Means alone hide preemption.',
        'Pin: engine version and commit, model revision, quantization scheme, KV dtype, attention backend, sampling parameters, seed, tokenizer, and GPU clock/power limit.',
        'Report whether prefix caching was enabled and the observed hit rate. A benchmark with an accidental 90% hit rate measures your cache, not your engine.',
        'Discard warmup. Capture CUDA graphs, compile, and run a few hundred requests before recording.',
        'Cross-check with a second, independently implemented tool. When two tools disagree, that disagreement is data — usually about how TTFT is defined or how streaming is parsed.',
        'State the failure criterion: at what load did you stop, and why. "Throughput at saturation" without a latency SLO is not a useful number.'
      ],
      tables: [{
        title: 'Benchmark pre-publication checklist',
        cols: ['Category', 'Must state'],
        rows: [
          ['Hardware', 'GPU model and count, interconnect, host CPU, power/clock limits'],
          ['Model', 'Exact repo + revision, dtype, quantization method, KV dtype'],
          ['Engine', 'Name, version, commit, all non-default flags, attention backend'],
          ['Workload', 'Input length, output length, distributions, prompt corpus or generator + seed'],
          ['Load', 'Concurrency levels swept, request rate model (Poisson vs closed loop), duration, warmup discarded'],
          ['Sampling', 'temperature, top_p, top_k, penalties, seed, whether ignore_eos was used'],
          ['Caching', 'Prefix caching on/off, measured hit rate'],
          ['Metrics', 'Definitions used for TTFT/ITL/TPOT, percentiles reported, how streaming was parsed'],
          ['Result', 'p50/p95/p99 TTFT and ITL, output tok/s, goodput under a stated SLO'],
          ['Repro', 'Exact commands, container image digest, raw result files']
        ]
      }],
      pitfalls: [
        'Using `ignore_eos=false` so output length varies with the model, making two models incomparable. Fix output length with `ignore_eos` for engine comparisons and say you did.',
        'Closed-loop load generation (N workers each waiting for a response) when your production traffic is open-loop (Poisson arrivals). Closed loop cannot produce queue buildup and will hide your worst behaviour.',
        'Running on a shared or virtualized GPU host with noisy neighbours.',
        'Reporting an average over a run whose queue was growing. The system was unstable; the number is meaningless.',
        'Comparing against a "HuggingFace Transformers baseline" configured naively. It flatters everything and teaches nothing.'
      ],
      build: [
        'Write your benchmark checklist into a template file in your repo and fill it in for every run you keep.',
        'Take any public inference benchmark blog post and grade it against the table above. Note how many rows it fails.'
      ],
      checks: [
        'Why does open-loop vs closed-loop load generation change your conclusions?',
        'Why must output length be fixed when comparing two engines?',
        'Two tools report TTFT differing by 80 ms on the same run. Name three plausible causes.'
      ],
      res: ['r-guidellm-intro', 'r-modular-benchmarks', 'r-vllm-bench-docs', 'r-mlperf-inference', 'r-artificial-analysis']
    },

    {
      id: 'u-instrumentation', tier: 2,
      title: 'Instrumentation: the metrics that actually diagnose',
      summary: 'Prometheus, Grafana, and the specific series that tell you what is wrong.',
      why: 'Latency percentiles tell you that something is wrong. Engine-internal metrics tell you what. The difference between "p99 TTFT is 8 seconds" and "p99 TTFT is 8 seconds because KV utilization hit 100% and we preempted 40 requests per minute" is the difference between guessing and fixing.\n\nModern engines expose a good set of internal metrics on a `/metrics` endpoint. Very few teams graph them. Build the dashboard before you need it, because the moment you need it is an incident.',
      core: [
        'The core five: TTFT histogram, ITL/TPOT histogram, output tokens/sec, `num_requests_running`, `num_requests_waiting`.',
        'The diagnostic five: KV cache utilization, preemption count, prefix cache hit rate, average batch size (running requests per step), and step time.',
        'Use histograms with sensible buckets, not gauges of the mean. You cannot recover percentiles from an average.',
        'Correlate at the request level: log request id, prompt tokens, output tokens, TTFT, E2E, and whether it was preempted. Aggregates hide per-request pathologies.',
        'Watch queue time separately from prefill time inside TTFT. They have completely different fixes.',
        'Cost per request belongs on the same dashboard as latency. If it is not visible, nobody optimizes it.',
        'Alert on goodput and on queue depth, not on GPU utilization.'
      ],
      code: [{
        title: 'Minimum viable observability stack',
        body: `# engine exposes /metrics in Prometheus format
vllm serve <model> --disable-log-requests   # keep metrics, drop noise

# prometheus.yml
scrape_configs:
  - job_name: vllm
    scrape_interval: 1s          # LLM steps are ~10-50ms; 15s hides everything
    static_configs:
      - targets: ['localhost:8000']

# Grafana panels to build, in this order:
#  1. TTFT p50/p95/p99          (histogram_quantile over the bucket series)
#  2. ITL  p50/p95/p99
#  3. output tokens/sec
#  4. running vs waiting requests (stacked)
#  5. KV cache usage %
#  6. preemptions/min
#  7. prefix cache hit rate
#  8. estimated $/1M tokens (derived: rate * $/hr / tokens_per_sec)`,
        note: 'Scrape interval matters. At the default 15 s you will average away every spike you were trying to see.'
      }],
      pitfalls: [
        'Scraping at 15 s and wondering why the dashboard is smooth while users complain.',
        'Graphing GPU utilization as a health metric. It is ~100% whenever anything is running.',
        'Computing percentiles client-side from a mean and a standard deviation.',
        'Losing the correlation between a slow request and the engine state at that moment, because logs and metrics are not joined by timestamp and request id.'
      ],
      build: [
        'Stand up Prometheus + Grafana against your engine and build the eight panels above. Put the dashboard JSON in your repo.',
        'Run a load test that induces preemption and confirm you can identify the exact moment and cause from the dashboard alone, without looking at logs.'
      ],
      checks: [
        'How do you decompose TTFT into queue time and prefill time from metrics?',
        'Which metric would first indicate that prefix caching regressed after a prompt template change?',
        'Why is a mean latency gauge nearly useless for this workload?'
      ],
      res: ['r-vllm-metrics-docs', 'r-guidellm-intro', 'r-modular-benchmarks', 'r-ml-engineering']
    },

    {
      id: 'u-loadtest', tier: 2,
      title: 'Load generation to saturation and beyond',
      summary: 'GuideLLM, genai-perf, vllm bench: sweeping properly and finding the knee.',
      why: 'A load test that runs at one concurrency level answers one question badly. What you want is the curve: throughput and latency percentiles as a function of offered load, from synchronous baseline through the knee to saturation and into overload. The knee — where throughput stops rising and latency starts climbing steeply — is the only interesting operating point, and it is the number you should design capacity around.\n\nUse two independently implemented tools. They will disagree, and reconciling the disagreement will teach you more about your definitions than any single run.',
      core: [
        'GuideLLM sweeps from synchronous to saturation automatically and reports the whole curve rather than a single point.',
        '`vllm bench serve` is the engine-native harness; useful as a cross-check and because it understands the engine\'s own metrics.',
        'genai-perf (from the Triton perf_analyzer family) does concurrency sweeps at 1, 2, 4 ... 128 and is a good third opinion.',
        'Open-loop (Poisson arrival) load reveals queueing behaviour; closed-loop (fixed concurrency) reveals steady-state throughput. Run both and report which is which.',
        'Fix input/output lengths for engine comparisons; use a realistic length distribution for capacity planning. These are two different experiments and should not be conflated.',
        'Push past the knee deliberately. You need to know your overload behaviour before production finds it for you.',
        'Record raw per-request results, not just summaries. You will want to recompute percentiles later with a different definition.'
      ],
      code: [{
        title: 'A sweep that is worth publishing',
        body: `# 1. Synchronous baseline (concurrency 1) - the latency floor
guidellm benchmark --target http://localhost:8000 \\
  --rate-type synchronous --data "prompt_tokens=1024,output_tokens=256"

# 2. Full sweep to saturation
guidellm benchmark --target http://localhost:8000 \\
  --rate-type sweep --max-seconds 120 \\
  --data "prompt_tokens=1024,output_tokens=256"

# 3. Engine-native cross-check at a fixed concurrency
vllm bench serve --model <model> --num-prompts 1000 \\
  --random-input-len 1024 --random-output-len 256 \\
  --max-concurrency 64 --ignore-eos --seed 0

# 4. Overload probe: 1.5x the measured knee, watch preemptions
#    and num_requests_waiting on the dashboard while it runs.`,
        note: 'Always include `--ignore-eos` (or the equivalent) when comparing engines, so output length is a controlled variable rather than a model behaviour.'
      }],
      pitfalls: [
        'Running the load generator on the same host as the server and attributing its CPU contention to the engine.',
        'Letting the client become the bottleneck. Verify the generator can produce the request rate it claims.',
        'A too-short run: 30 seconds at 20 req/s is 600 samples, which cannot estimate a p99.',
        'Reusing the same prompt for every request, which turns on prefix caching and measures the cache.',
        'Ignoring connection limits and ephemeral port exhaustion at high concurrency.'
      ],
      build: [
        'Produce the full sweep curve for your service. Mark the knee, the goodput ceiling under a stated SLO, and the point of instability.',
        'Cross-check one operating point with a second tool. Explain any discrepancy larger than 10% in terms of metric definitions.'
      ],
      checks: [
        'How do you tell from the results that a run was unstable and must be discarded?',
        'Why is `--ignore-eos` important for engine comparison and wrong for capacity planning?',
        'How many samples do you need for a trustworthy p99?'
      ],
      res: ['r-guidellm', 'r-guidellm-intro', 'r-vllm-bench-docs', 'r-genai-perf', 'r-inference-benchmarker']
    },

    {
      id: 'u-profiling', tier: 3,
      title: 'Profiling down to the kernel',
      summary: 'Timeline traces, Nsight, and attributing milliseconds to specific causes.',
      why: 'When the aggregate numbers say "slow", profiling says "this kernel, this gap, this reason". Two tools cover almost everything: a timeline profile (PyTorch profiler or Nsight Systems) to find where time goes at the step level, and a kernel profile (Nsight Compute) to find why a specific kernel is slow.\n\nThe distinction matters. Timeline profiling finds gaps, serialization, and unexpected synchronizations — the overhead-bound regime. Kernel profiling finds low occupancy, bad memory access patterns, and roofline placement — the compute and bandwidth regimes. Reaching for the wrong one wastes an afternoon.',
      core: [
        'Start with a timeline. Look for: gaps between kernels (launch overhead), long single kernels (candidates for optimization), host-device synchronizations, and CPU-side blocking.',
        'Only after the timeline is clean should you profile individual kernels. Optimizing a kernel that occupies 4% of the step is a waste.',
        'Nsight Compute gives you the roofline directly, plus memory throughput and occupancy. Use its roofline section to confirm your hand analysis from Module 00.',
        'Capture only a few steps. Profilers add overhead and produce enormous traces; ten decode steps is usually plenty.',
        'Attribute by phase: prefill steps and decode steps have completely different profiles and must be captured separately.',
        'The CPU side matters as much as the GPU side in serving. Profile the scheduler process, not just the model.',
        'A trace viewer (Perfetto, Chrome tracing) is the fastest way to see a serialization bug that no counter will reveal.'
      ],
      code: [{
        title: 'Capturing useful traces',
        body: `# PyTorch profiler around N steps (engine-integrated in vLLM via
# VLLM_TORCH_PROFILER_DIR, then start/stop endpoints)
export VLLM_TORCH_PROFILER_DIR=/tmp/vllm_trace
vllm serve <model> ...
curl -X POST localhost:8000/start_profile
#  ... send exactly the traffic you care about ...
curl -X POST localhost:8000/stop_profile
# open the trace in https://ui.perfetto.dev

# Nsight Systems: whole-timeline, low overhead
nsys profile -t cuda,nvtx,osrt -o decode_trace --force-overwrite true \\
  python bench_client.py

# Nsight Compute: one kernel, full detail (high overhead, few launches)
ncu --set full --kernel-name-base demangled \\
    -k regex:"flash|attn" -c 5 -o attn_profile python bench_step.py`,
        note: 'Questions to answer from the trace: what fraction of the step is gaps? which kernel is the largest single contributor? is there a device sync per step? does the CPU finish scheduling before the GPU finishes the previous step?'
      }],
      pitfalls: [
        'Profiling with the profiler\'s overhead included in your headline number.',
        'Capturing a trace that includes warmup and graph capture and drawing conclusions from it.',
        'Using Nsight Compute on a full serving run. It serializes kernels and can slow things 100x.',
        'Optimizing the kernel with the worst efficiency rather than the kernel with the most total time.'
      ],
      build: [
        'Capture a decode-step trace and produce a breakdown: attention %, MLP GEMM %, LM head %, sampling %, gaps %. Numbers must sum to 100.',
        'Run the Nsight Compute roofline on one attention kernel and compare its measured arithmetic intensity to your hand calculation from Module 00.'
      ],
      checks: [
        'When do you reach for a timeline profiler versus a kernel profiler?',
        'Your trace shows 30% gaps. List the three most likely causes in order.',
        'Why should prefill and decode be profiled separately?'
      ],
      res: ['r-gpu-mode-l1', 'r-nsight-docs', 'r-pytorch-profiler', 'r-brrr', 'r-vllm-profiling-docs']
    },

    {
      id: 'u-quality-eval', tier: 2,
      title: 'Measuring quality, because every optimization spends some',
      summary: 'Quantization, eviction and speculation all have a quality budget. Measure it or you are guessing.',
      why: 'Almost every technique in Modules 05-07 trades a small amount of output quality for a large amount of speed or capacity — except speculative decoding, which is provably distribution-preserving, and that exception is exactly why it is valuable. If you do not measure quality, you cannot tell a 2x speedup that is free from a 2x speedup that quietly broke long-context retrieval.\n\nThe practical requirement is modest: one fast, deterministic, task-relevant eval you can run on every variant in under ten minutes, plus one long-context stress test. Perplexity alone is not sufficient and is often actively misleading for KV-cache optimizations.',
      core: [
        'Build a fixed eval set from your own traffic distribution. Public benchmarks measure general capability, not your task.',
        'Perplexity is a weak proxy: it is dominated by local context and barely moves when long-range retrieval breaks.',
        'Always include a long-context retrieval test (needle-in-a-haystack style) when evaluating KV quantization or eviction.',
        'Use greedy decoding and a fixed seed for quality comparisons, so that sampling noise does not swamp the effect.',
        'Report the metric with a confidence interval or an n. A 1% difference on 200 examples is noise.',
        'Speculative decoding with proper rejection sampling produces the *same distribution* as the target model. If your spec-decode variant scores differently beyond noise, you have a bug, not a trade-off.',
        'Keep the eval in CI. Quality regressions from an engine upgrade or a flag change are common and silent.'
      ],
      tables: [{
        title: 'What each optimization spends',
        cols: ['Technique', 'Quality cost', 'What to test'],
        rows: [
          ['FP8 weights (per-channel)', 'Usually negligible', 'Task eval; check outlier-heavy prompts'],
          ['INT4 weight-only (AWQ/GPTQ)', 'Small but real, worse on small models', 'Task eval + reasoning/math subset'],
          ['W8A8 / SmoothQuant', 'Small; activation outliers are the risk', 'Task eval on long, unusual inputs'],
          ['FP8 KV cache', 'Small; concentrated in long context', 'Long-context retrieval'],
          ['INT4 KV cache', 'Noticeable without per-channel keys', 'Long-context retrieval + math'],
          ['KV eviction (H2O/SnapKV)', 'Workload dependent, can be severe', 'Needle-in-haystack, multi-hop'],
          ['Sliding window without sinks', 'Severe', 'Anything long-range'],
          ['Speculative decoding', 'None, by construction', 'Verify distribution match, then ignore'],
          ['Prefix caching', 'None', 'Correctness only']
        ]
      }],
      pitfalls: [
        'Evaluating with sampling on and attributing the variance to the optimization.',
        'Using perplexity for KV cache changes.',
        'Testing quality at 2k context and deploying at 128k.',
        'Comparing an FP16 baseline run on a different engine version than the quantized variant.'
      ],
      build: [
        'Assemble a 200-500 example eval from your own domain, with a deterministic scorer. Time it; it must run in under ten minutes.',
        'Score FP16, FP8, and INT4 variants of the same model plus one KV-quantized variant. Produce the single table you will use for every future decision.'
      ],
      checks: [
        'Why is perplexity a poor metric for KV cache eviction?',
        'Why should speculative decoding show zero quality change, and what does it mean if it does not?',
        'How many examples do you need to detect a 2% accuracy regression?'
      ],
      res: ['r-lm-eval-harness', 'r-needle-haystack', 'r-kbit-scaling-laws', 'r-modular-benchmarks']
    }
  ]
},

/* ==================================================================== 05 */
{
  id: 'm05', num: '05',
  title: 'Quantization',
  tagline: 'Number formats, weight-only PTQ, activation quantization, and the FP8/FP4 era.',
  focus: 'Quantization is the highest-leverage single change available to a decode-bound service, because it attacks bytes rather than FLOPs. This module builds from what the number formats actually are, through the two families of post-training quantization (weight-only and weight+activation), to what production stacks run today and how to measure the damage honestly.',
  outcome: 'FP8 and INT4 variants of your model, benchmarked against the FP16 baseline on the same sweep, with a quality table that shows what each cost.',
  units: [
    {
      id: 'u-numerics', tier: 2,
      title: 'Number formats: FP16, BF16, FP8, INT8, INT4, MXFP4, NVFP4',
      summary: 'Exponent bits, mantissa bits, scaling granularity, and what each format actually buys.',
      why: 'Nearly every quantization confusion is a numerics confusion. BF16 and FP16 have the same size but very different dynamic range, which is why one needs loss scaling and the other does not. FP8 comes in two flavours whose difference (E4M3 vs E5M2) decides whether you use it for weights or gradients. INT4 is not one thing: it is a family defined by the scaling granularity, and group-wise INT4 with a per-128 scale is a different algorithm from per-tensor INT4.\n\nThe new microscaling formats (MXFP4, NVFP4) exist because per-tensor scaling cannot handle outliers and per-element scaling costs as much as the data. A shared exponent over a small block is the compromise, and it is now built into hardware.',
      core: [
        'FP16: 1-5-10. High precision, narrow range; overflow risk. BF16: 1-8-7. Same range as FP32, less mantissa; the training default and a safe inference default.',
        'FP8 E4M3: 1-4-3, range ±448, used for weights and activations in inference. FP8 E5M2: 1-5-2, wider range, less precision, used where dynamic range matters more.',
        'INT8: uniform, symmetric or asymmetric, with a scale (and possibly zero point). Cheap and well-supported; needs outlier handling for activations.',
        'INT4 / NF4: 16 levels. Weight-only in practice. NF4 (from QLoRA) uses information-theoretically optimal levels for normally distributed weights rather than uniform ones.',
        'Scaling granularity is the real design axis: per-tensor (one scale) < per-channel (one per output channel) < group-wise (one per 32/64/128 elements) < per-element. Finer granularity costs metadata and kernel complexity but handles outliers.',
        'MXFP4 / NVFP4: block floating point with a shared exponent per small block (32 for MX, 16 for NVFP4 with an FP8 scale). Hardware-supported on Blackwell; this is where FP4 inference becomes practical rather than academic.',
        'The memory saving is exact and predictable. The *speed* saving depends entirely on whether a fused kernel exists for that format on your GPU. Format support and kernel support are different questions.'
      ],
      tables: [{
        title: 'Formats at a glance',
        cols: ['Format', 'Bits', 'Layout', 'Typical use', 'Notes'],
        rows: [
          ['FP32', '32', '1-8-23', 'Accumulation, master weights', 'Never for serving weights'],
          ['BF16', '16', '1-8-7', 'Default serving dtype', 'FP32 range, no loss scaling needed'],
          ['FP16', '16', '1-5-10', 'Serving on older GPUs', 'Overflow risk in attention accumulation'],
          ['FP8 E4M3', '8', '1-4-3', 'Weights + activations', '±448 range; needs per-tensor or finer scales'],
          ['FP8 E5M2', '8', '1-5-2', 'Wide-range tensors', 'Less mantissa; rarely used for weights'],
          ['INT8', '8', 'integer + scale', 'W8A8 (SmoothQuant), KV cache', 'Activation outliers are the problem'],
          ['INT4 / GPTQ / AWQ', '4', 'integer + group scale', 'Weight-only PTQ', 'Group size 128 is the common default'],
          ['NF4', '4', 'non-uniform levels', 'QLoRA, memory-bound serving', 'Optimal for Gaussian weights'],
          ['MXFP4', '4', 'E2M1 + shared E8M0 per 32', 'Blackwell-class inference', 'Open OCP standard'],
          ['NVFP4', '4', 'E2M1 + FP8 scale per 16', 'Blackwell inference', 'Finer blocks, better accuracy than MXFP4']
        ]
      }],
      math: [{
        title: 'Quantization, exactly',
        body: `Symmetric uniform quantization to b bits:
    s   = max(|W_group|) / (2^(b-1) - 1)
    W_q = round(W / s)                clamp to [-2^(b-1), 2^(b-1)-1]
    W'  = W_q * s

Asymmetric adds a zero point:
    s = (max - min) / (2^b - 1),  z = round(-min / s)
    W_q = round(W/s) + z

Metadata overhead for group size g, scale in FP16:
    bits/weight = b + 16/g        (+16/g again if zero points)
    b=4, g=128 -> 4.125 bits/weight  (4.25 with zero points)

Memory for a 70B model:
    BF16      : 140.0 GB
    FP8       :  70.0 GB
    INT4 g128 :  36.1 GB   (not 35.0 - the scales are real)`,
        note: 'Always include the metadata in your memory budget. "4-bit" is 4.125-4.5 bits in practice, and on a 70B model that difference is over a gigabyte.'
      }],
      pitfalls: [
        'Assuming FP8 support means fast FP8. Ada and Hopper have FP8 tensor cores; Ampere does not, and an "FP8" checkpoint there is dequantized to BF16 on the fly.',
        'Ignoring scale metadata in memory calculations.',
        'Using FP16 accumulation in attention on long sequences; overflow is real and shows up as NaNs at scale.',
        'Confusing storage format with compute format. W4A16 stores 4-bit and computes in 16-bit after dequantization; it saves bandwidth, not FLOPs.'
      ],
      build: [
        'For one model, tabulate exact memory at BF16, FP8, INT8, INT4-g128 and INT4-g32 including metadata. Note where the group-size trade stops being worth it.',
        'Check which of those formats have fused kernels on your GPU in your engine. Record which combinations are actually fast, not merely supported.'
      ],
      checks: [
        'Why does BF16 not need loss scaling while FP16 does?',
        'Compute the true bits per weight for INT4 with group size 64, FP16 scales and INT4 zero points.',
        'What is the difference between MXFP4 and NVFP4, and why does block size matter?'
      ],
      res: ['r-fp8-formats', 'r-qlora', 'r-mxfp4-spec', 'r-mit-6s965', 'r-lilian-inference']
    },

    {
      id: 'u-weight-ptq', tier: 2,
      title: 'Weight-only PTQ: GPTQ and AWQ',
      summary: 'The two algorithms behind almost every 4-bit checkpoint you will ever serve.',
      why: 'Weight-only quantization is the highest-value option for decode-bound serving: decode reads weights every step, so halving or quartering weight bytes directly cuts step time, while activations stay in BF16 so accuracy is easy to preserve. The question is only how to choose the 4-bit values.\n\nGPTQ answers it with second-order information: quantize weights column by column, and after each one, update the remaining weights to compensate using an approximation of the Hessian of the layer\'s reconstruction error. AWQ answers it differently: observe that a small fraction of weight channels matter far more (identified via *activation* magnitude, not weight magnitude), and scale those channels up before quantization so they land on finer effective grid points. Both are one-shot, calibration-set based, and take minutes to hours. Knowing which assumption each makes tells you when each fails.',
      core: [
        'GPTQ: layer-wise reconstruction with OBQ-style error compensation using an inverse Hessian estimate from a calibration set. Quantizes 175B models in hours on one GPU.',
        'AWQ: protects salient weight channels identified by activation magnitude, applies a per-channel scaling before quantization so that no weights need to be kept in higher precision. Better generalization to instruction-tuned and multimodal models; MLSys 2024 best paper.',
        'Both need a calibration set. Its distribution matters: calibrating on wikitext and serving code is a known way to lose accuracy on code.',
        'Group size trades accuracy against metadata. 128 is the default; 64 or 32 helps damaged models at real memory cost.',
        'Activation-aware methods degrade more gracefully on small models. Quantization damage grows as model size shrinks — a 70B at 4 bits often beats an 8B at 16 bits per unit of memory, which is the practical content of the k-bit scaling laws work.',
        'Marlin and similar kernels make W4A16 actually fast; without a good kernel, dequantization overhead can eat the bandwidth win at high batch.',
        'Modern production stacks increasingly use `llm-compressor` / compressed-tensors to produce checkpoints and vLLM to serve them; the algorithm is a flag, the calibration data is the thing you own.'
      ],
      math: [{
        title: 'What each method optimizes',
        body: `GPTQ - minimize layer output error, weight by weight:
    argmin_Wq || W X - Wq X ||_2^2
    Solved greedily with error feedback:
      delta = (w_q - w) / [H^-1]_qq
      W_remaining -= delta * H^-1[q, remaining]
    H = 2 X X^T  (from calibration activations)

AWQ - scale before quantizing, chosen per channel:
    Q(W * diag(s)) * diag(s)^-1 @ X
    s chosen to minimize output error; salient channels
    (large |X| statistics) get larger s -> finer effective grid.

Key difference: GPTQ compensates AFTER quantizing;
AWQ conditions the problem BEFORE quantizing.`,
        note: 'Both are trying to preserve `WX`, not `W`. That is the central insight of activation-aware quantization: weight magnitude is the wrong saliency signal.'
      }],
      pitfalls: [
        'Calibrating on a corpus unlike your traffic. This is the number one cause of "the quantized model got dumber at my task".',
        'Comparing a GPTQ checkpoint from one author against an AWQ checkpoint from another and concluding something about the algorithms. Group size, calibration data and kernel all differ.',
        'Quantizing the LM head or embeddings without checking. Some recipes exclude them for good reason.',
        'Expecting a throughput win at very large batch. W4A16 helps most when weight bandwidth dominates, i.e. small to moderate batch.'
      ],
      build: [
        'Produce an AWQ or GPTQ 4-bit checkpoint of your model with a calibration set drawn from your own traffic. Serve it. Run the same sweep as your FP16 baseline.',
        'Run your quality eval on both. Produce a table with memory, throughput at three concurrency levels, and quality. Decide, and write down why.'
      ],
      checks: [
        'Why is activation magnitude a better saliency signal than weight magnitude?',
        'When does W4A16 stop helping throughput as batch size grows?',
        'What exactly does GPTQ\'s Hessian approximate, and what breaks if the calibration set is unrepresentative?'
      ],
      res: ['r-gptq', 'r-awq', 'r-awq-repo', 'r-kbit-scaling-laws', 'r-marlin', 'r-llm-compressor', 'r-mit-6s965']
    },

    {
      id: 'u-act-quant', tier: 3,
      title: 'Activation quantization: outliers, SmoothQuant, rotations',
      summary: 'Why W8A8 is harder than W8A16, and the three generations of fixes.',
      why: 'Weight-only quantization saves bandwidth but leaves compute in BF16. To actually use the FP8/INT8 tensor cores — which matters for prefill and for large-batch decode — activations must be quantized too, and activations are much harder. Transformer activations contain systematic outliers concentrated in a small number of channels, tens to hundreds of times larger than the rest, and a per-tensor scale that accommodates them destroys precision for everything else.\n\nThe fixes form a clear progression: keep the outliers in high precision (LLM.int8()), migrate the difficulty from activations to weights (SmoothQuant), or rotate the representation so outliers disappear entirely (QuaRot, QServe). Each is a different answer to the same structural fact, and knowing the fact is what lets you predict which model will quantize badly.',
      core: [
        'Activation outliers appear in specific channels consistently across tokens, emerge above roughly 6.7B parameters, and are load-bearing for accuracy — you cannot simply clip them.',
        'LLM.int8(): decompose the matmul, keep outlier channels in FP16 and the rest in INT8. Correct but the mixed-precision path limits speed.',
        'SmoothQuant: migrate quantization difficulty from activations to weights with a per-channel scale `diag(s)^-1` folded into the previous layer, since weights are far easier to quantize. Enables true W8A8 with negligible loss.',
        'QuaRot / rotation-based methods: apply Hadamard/orthogonal rotations that provably spread outlier energy across channels, making both weights and activations quantizable — the current state of the art for aggressive W4A4/W4A8.',
        'QServe (W4A8KV4) shows the practical shape of the modern answer: 4-bit weights, 8-bit activations, 4-bit KV, with careful attention to where dequantization happens in the kernel.',
        'Per-token dynamic activation scales are usually necessary and cheap; per-tensor static scales are faster but need calibration and are more fragile.',
        'FP8 activations are much more forgiving than INT8 because the exponent gives dynamic range for free. This is why FP8 W8A8 shipped broadly before INT8 W8A8 did.'
      ],
      math: [{
        title: 'SmoothQuant\'s migration',
        body: `Y = X W                (X has outlier channels, W does not)

Insert a diagonal scaling that cancels:
    Y = (X diag(s)^-1) (diag(s) W)
        \\_____ X' _____/  \\____ W' ____/

Choose per-channel:
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)

alpha = 0.5 splits the difficulty evenly.
alpha -> 1 pushes all difficulty into W; alpha -> 0 leaves it in X.

diag(s)^-1 folds into the preceding LayerNorm/projection,
so it is free at inference time.`,
        note: 'The elegance is that the scaling is mathematically an identity — it changes nothing about the function, only the numerical conditioning of the quantization problem.'
      }],
      pitfalls: [
        'Applying W8A8 to a model whose activation outliers were never measured. Profile the activation ranges per channel first.',
        'Static activation scales calibrated on short prompts, then serving long ones with different activation statistics.',
        'Assuming FP8 activations need the same care as INT8. They need less, but not none.',
        'Quantizing activations into attention without checking the softmax accumulator precision.'
      ],
      build: [
        'Dump per-channel activation maxima for a few layers on real prompts and plot the distribution. See the outliers with your own eyes.',
        'Serve an FP8 W8A8 checkpoint and compare prefill throughput against BF16. This is where activation quantization pays; decode at small batch will show much less.'
      ],
      checks: [
        'Why do activation outliers make per-tensor INT8 fail while per-channel weights are fine?',
        'Explain why SmoothQuant\'s scaling is free at inference time.',
        'Why does a rotation make a tensor easier to quantize?'
      ],
      res: ['r-llm-int8', 'r-smoothquant', 'r-quarot', 'r-qserve', 'r-mit-6s965', 'r-zeroquant']
    },

    {
      id: 'u-fp8-prod', tier: 2,
      title: 'FP8 and FP4 in production',
      summary: 'What actually ships: recipes, kernels, hardware support, and where the wins really come from.',
      why: 'Between the papers and production there is a gap filled with kernel availability, checkpoint formats, and hardware generations. FP8 on Hopper is a mature, well-supported path with near-free quality; FP4 on Blackwell is the current frontier and is where the next 2x is coming from. Knowing what combination of (GPU, format, engine, kernel) is fast today is worth more than knowing another algorithm.\n\nThis unit is deliberately practical: what to enable, what to measure, and what the honest expected gains are for prefill, for decode at low batch, and for decode at high batch — three numbers that are frequently conflated into one misleading claim.',
      core: [
        'FP8 weights + FP8 activations doubles tensor core throughput on Hopper/Ada and halves weight bytes. Prefill gains are large; decode gains at small batch come mostly from the bandwidth half.',
        'FP8 KV cache is a separate flag with separate gains, discussed in Module 02. Enable and measure independently.',
        'Blackwell adds FP4 tensor cores with hardware block scaling (NVFP4). Reported gains are substantial for both prefill and large-batch decode, and the accuracy story is much better than software FP4 was.',
        'Checkpoint ecosystems: `compressed-tensors` (llm-compressor, vLLM), TensorRT-LLM\'s quantization toolkit, and AutoAWQ/AutoGPTQ formats. They are not interchangeable; check what your engine loads.',
        'Expected gains, honestly: prefill ~1.5-1.8x for FP8 over BF16; decode at batch 1 ~1.6-1.9x (bandwidth); decode at large batch approaches the compute ratio. Anyone quoting a single number is hiding one of these.',
        'Mixed recipes are normal: FP8 weights + BF16 activations for sensitive layers, FP8 KV, INT4 for the largest MLP matrices. Production configs are rarely uniform.',
        'Always re-tune after quantizing. The optimal batch size, token budget and CUDA graph shapes move because the roofline position moved.'
      ],
      tables: [{
        title: 'Format support by GPU generation',
        cols: ['GPU', 'BF16', 'FP8 tensor cores', 'FP4 tensor cores', 'Practical guidance'],
        rows: [
          ['A100 (Ampere)', 'Yes', 'No', 'No', 'W4A16 weight-only is the main lever'],
          ['L40S / Ada', 'Yes', 'Yes', 'No', 'FP8 works; bandwidth-limited for decode'],
          ['H100 / H200 (Hopper)', 'Yes', 'Yes', 'No', 'FP8 W8A8 + FP8 KV is the default recipe'],
          ['B200 / GB200 (Blackwell)', 'Yes', 'Yes', 'Yes (NVFP4)', 'NVFP4 for weights and activations where quality allows'],
          ['MI300X (CDNA3)', 'Yes', 'Yes (FP8)', 'No', 'Check kernel coverage in your engine, it lags']
        ]
      }],
      pitfalls: [
        'Enabling FP8 on Ampere and measuring a slowdown, because the weights are dequantized to BF16 every step.',
        'Reporting the prefill speedup as "the FP8 speedup" for a decode-bound service.',
        'Loading a checkpoint quantized with a different toolkit and getting silent accuracy loss from a scale-format mismatch.',
        'Forgetting to re-tune batch and token budget after the roofline position changed.'
      ],
      build: [
        'Produce FP8 and (if available) FP4 variants. Run the identical sweep as the BF16 baseline. Report three separate speedups: prefill, decode@1, decode@64.',
        'Run the quality eval on each. Build the final decision table: memory, three speedups, quality delta, and cost per million tokens.'
      ],
      checks: [
        'Why is the FP8 speedup different for prefill, low-batch decode, and high-batch decode?',
        'What does hardware block scaling in NVFP4 solve that software FP4 could not?',
        'What must be re-tuned after switching precision, and why?'
      ],
      res: ['r-vllm-quant-docs', 'r-llm-compressor', 'r-nvidia-inference-guide', 'r-trtllm-repo', 'r-qserve', 'r-modular-handbook']
    }
  ]
},

/* ==================================================================== 06 */
{
  id: 'm06', num: '06',
  title: 'Decoding algorithms',
  tagline: 'Speculative decoding, draft models, parallel decoding, and constrained generation.',
  focus: 'Decode is memory bound, which means FLOPs are nearly free at low batch. Speculative decoding is the systematic exploitation of that fact: spend cheap parallel compute to reduce the number of sequential memory passes. This module derives the algorithm and its guarantee, surveys the draft mechanisms, and — most importantly — establishes the regime where it stops working, because that regime is where most production traffic lives.',
  outcome: 'A speculative decoding deployment benchmarked at low and high QPS with the crossover point identified, including the negative result, plus a working constrained-decoding path.',
  units: [
    {
      id: 'u-spec-decode', tier: 2,
      title: 'Speculative decoding: the algorithm and its guarantee',
      summary: 'Draft cheaply, verify in parallel, accept with a rejection rule that provably preserves the target distribution.',
      why: 'Speculative decoding is the rare optimization with no quality cost, and the proof is worth understanding rather than taking on faith. A small draft model proposes `gamma` tokens; the target model scores all of them in a single forward pass — which costs almost the same as scoring one token, because decode is memory bound — and a modified rejection sampling rule accepts a prefix of them. The rule is constructed so that the accepted tokens are distributed exactly as if they had been sampled from the target model one at a time.\n\nThat guarantee is what makes it deployable without an eval. The catch is entirely economic: you pay `gamma` draft forward passes and some verification overhead for an expected number of accepted tokens that depends on how well the draft agrees with the target. When the GPU is already saturated by a large batch, the "free FLOPs" premise collapses and the whole thing becomes a loss.',
      core: [
        'Draft `gamma` tokens autoregressively with a cheap model `q`. Score all `gamma+1` positions with the target `p` in one pass.',
        'Accept token `i` with probability `min(1, p(x_i)/q(x_i))`. On rejection, resample from the normalized residual `norm(max(0, p - q))` and stop. This yields exactly the target distribution — this is the theorem.',
        'Expected accepted tokens per iteration is `(1 - alpha^(gamma+1)) / (1 - alpha)` where `alpha` is the average per-token acceptance rate.',
        'Speedup is that expectation divided by the cost of one iteration relative to one normal step: roughly `1 + c*gamma + overhead`, with `c` the draft-to-target cost ratio.',
        'There is an optimal `gamma`. Past it, drafting cost grows linearly while expected accepted tokens saturates at `1/(1-alpha)`.',
        'Acceptance rate is workload dependent: high on structured/predictable text (code, JSON, repeated context), low on creative or high-entropy text. Measure it on your traffic.',
        'At high batch, the target forward pass is no longer memory bound, so verifying `gamma+1` positions costs `gamma+1` times as much. The premise fails and speculative decoding loses. vLLM measured up to 2.8x at QPS 1 and 1.4-1.8x *slowdown* at high QPS — the crossover, not the peak, is the finding.'
      ],
      math: [{
        title: 'Acceptance, expectation, and speedup',
        body: `Rejection rule (Leviathan et al. 2023):
  accept x ~ q  with prob  min(1, p(x)/q(x))
  on reject     x ~ norm(max(0, p - q))
  => output distribution == p, exactly.

Expected accepted tokens per iteration, acceptance rate alpha:
  E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)

Cost of one speculative iteration, in target-step units:
  C = 1 + c*gamma + v          c = draft/target cost, v = verify overhead

Speedup = E[tokens] / C

alpha=0.7, gamma=4, c=0.15, v=0.1:
  E = (1 - 0.7^5)/0.3 = (1-0.168)/0.3 = 2.77
  C = 1 + 0.6 + 0.1 = 1.7
  S = 1.63x

alpha=0.5, gamma=4, c=0.3, v=0.1:
  E = (1-0.031)/0.5 = 1.94 ; C = 1 + 1.2 + 0.1 = 2.3 -> S = 0.84x  (a LOSS)`,
        note: 'Use the calculator to explore this surface. The two variables that decide everything are alpha (a property of your draft/target pair and your traffic) and the effective batch size (which decides whether verification is really free).'
      }],
      pitfalls: [
        'Enabling speculative decoding globally and measuring aggregate throughput at high load. You will see a regression and may wrongly conclude the technique is bad.',
        'Reporting a speedup without reporting the measured acceptance rate. The speedup is not transferable without it.',
        'Using a draft model with a different tokenizer. It must share the vocabulary or you need a translation layer that costs more than it saves.',
        'Assuming the guarantee still holds under an approximate acceptance rule (some "lossy" variants trade the guarantee for higher acceptance — that is a quality decision, not a free one).'
      ],
      build: [
        'Derive the expected-tokens formula yourself from the geometric structure of the acceptance process before reading it.',
        'Enable speculative decoding with an n-gram or small draft model. Measure acceptance rate, ITL and throughput at QPS 1, 5, 20, 50. Plot speedup against QPS and mark the crossover.'
      ],
      checks: [
        'Prove that the accept/resample rule yields exactly the target distribution.',
        'At alpha = 0.8 and c = 0.1, what gamma maximizes speedup, and what is the ceiling as gamma grows?',
        'Explain mechanically why speculative decoding degrades at high batch.'
      ],
      res: ['r-spec-decode', 'r-spec-sampling', 'r-vllm-spec-blog', 'r-gpu-mode-l22', 'r-vllm-spec-docs', 'r-modular-spec']
    },

    {
      id: 'u-draft-models', tier: 3,
      title: 'Drafting mechanisms: n-gram, Medusa, EAGLE, MTP',
      summary: 'Five ways to produce candidate tokens cheaply, and the acceptance/cost trade of each.',
      why: 'The draft is where all the engineering is. A separate small model is conceptually simple but adds a second set of weights, a second KV cache and a second scheduler client. Medusa adds parallel prediction heads to the target model itself. EAGLE drafts in feature space rather than token space and reaches much higher acceptance. N-gram/prompt-lookup drafting needs no model at all and works remarkably well when the output copies from the input. Multi-token prediction bakes drafting into pretraining.\n\nEach makes a different trade between acceptance rate, draft cost, memory, and operational complexity, and the right answer depends heavily on workload. For RAG and code editing, prompt lookup is often unbeatable and free. For open-ended chat, EAGLE-class drafters win.',
      core: [
        'Independent draft model: a smaller model from the same family (e.g. 1B drafting for 70B). Simple, but `c` is often 3-10% and acceptance is moderate; needs identical tokenizer.',
        'N-gram / prompt lookup: propose continuations by matching the recent suffix against the prompt. Zero model cost, near-zero memory, high acceptance on summarization, RAG, code editing, and anything that quotes its input.',
        'Medusa: extra decoding heads on the target predict tokens at positions `t+1..t+k` in parallel, verified with a tree attention mask. No separate model, but heads must be trained and acceptance is lower than EAGLE.',
        'EAGLE 1/2/3: autoregress on the target\'s *hidden features* rather than tokens, with a small head. Much higher acceptance for very low cost; EAGLE-2/3 add dynamic draft trees. Currently the strongest general-purpose drafter.',
        'MTP (multi-token prediction, as in DeepSeek-V3): the model is pretrained with additional prediction heads, so drafting is native and acceptance is high. Only available if the model was trained for it.',
        'Tree drafting: propose a tree of candidates rather than a chain and verify all branches with one masked attention pass. Raises expected accepted tokens per iteration at modest extra cost.',
        'Whatever you pick, the operational cost matters: a second model doubles your deployment surface, memory, and failure modes.'
      ],
      tables: [{
        title: 'Drafting mechanisms compared',
        cols: ['Method', 'Draft cost c', 'Typical acceptance', 'Extra memory', 'Best for'],
        rows: [
          ['N-gram / prompt lookup', '~0', 'High on copy-heavy, ~0 otherwise', 'None', 'RAG, summarization, code edit'],
          ['Small draft model', '0.05-0.15', 'Moderate', 'Full second model + KV', 'General, simple to reason about'],
          ['Medusa', '~0.05', 'Moderate', 'Extra heads', 'No second model wanted'],
          ['EAGLE-2 / EAGLE-3', '0.02-0.10', 'High', 'Small head', 'General chat, best default'],
          ['MTP (native)', '~0.03', 'High', 'None (built in)', 'Models pretrained with MTP'],
          ['Tree drafting (any of above)', 'Higher', 'Higher E[tokens]', 'Tree buffers', 'Low-QPS latency-critical']
        ]
      }],
      pitfalls: [
        'Choosing a drafter without measuring acceptance on your own traffic. The published numbers are on their benchmarks, not yours.',
        'Deploying an untrained Medusa/EAGLE head. These require training on data matching the target model\'s distribution.',
        'Running prompt-lookup drafting on chat traffic with no input copying and being surprised that nothing is accepted.',
        'Ignoring the second model\'s memory when sizing KV cache; the draft KV is real.'
      ],
      build: [
        'Measure acceptance rate for n-gram drafting on two workloads: a RAG/summarization set and an open-ended chat set. The gap will be dramatic.',
        'If your engine supports EAGLE, compare it to n-gram on both workloads at QPS 1 and QPS 20. Four numbers, one table.'
      ],
      checks: [
        'Why does drafting in feature space (EAGLE) achieve higher acceptance than drafting in token space at similar cost?',
        'Why is prompt lookup nearly free, and on which workloads does it collapse?',
        'How does tree drafting change the expected-tokens formula?'
      ],
      res: ['r-medusa', 'r-eagle', 'r-eagle2', 'r-lookahead', 'r-specinfer', 'r-deepseek-v3', 'r-speculators']
    },

    {
      id: 'u-spec-prod', tier: 3,
      title: 'Speculative decoding in production',
      summary: 'Dynamic gamma, per-request decisions, and integration with batching and scheduling.',
      why: 'Speculative decoding is not a flag you turn on; it is a policy. The optimal `gamma` depends on the current batch size, the request\'s recent acceptance history, and the latency target. A static configuration is optimal at exactly one operating point and wrong everywhere else, which is why naive deployments show a win in the benchmark and a regression in production.\n\nThe production form of this technique is adaptive: reduce or disable speculation as batch size grows, tune `gamma` per request from a running acceptance estimate, and account for the fact that speculation changes the scheduler\'s token budget accounting. Getting this right is what converts a paper result into a service improvement.',
      core: [
        'Batch-adaptive speculation: disable or shrink `gamma` above a batch-size threshold. The threshold is where verification stops being free, and it is measurable.',
        'Per-request adaptive `gamma`: maintain an exponential moving average of acceptance and set `gamma` from it. Requests in a predictable region get more speculation.',
        'Speculation changes the scheduler\'s accounting: each speculative step consumes `gamma+1` token slots in the batch. A budget-based scheduler must be told, or it will over-admit.',
        'Interaction with prefix caching is positive: cached prefixes reduce prefill so more of the step is decode, where speculation helps.',
        'Interaction with chunked prefill is negative-ish: steps containing prefill chunks are already compute bound, so speculation on those steps is wasted.',
        'Latency SLOs change the calculus: at low QPS with a strict TTFT/ITL target, speculation is nearly always worth it. At high QPS with a throughput target, it usually is not.',
        'Measure and export acceptance rate as a first-class metric. A drop in acceptance is an early signal that traffic distribution shifted.'
      ],
      math: [{
        title: 'The batch crossover',
        body: `Verification of gamma+1 positions costs, per step:
  memory-bound regime : ~1x    (weights dominate, positions are free)
  compute-bound regime: ~(gamma+1)x

Effective batch during verification = B * (gamma+1)

Speculation is worth it while:
  B * (gamma+1) < B*        (B* = critical batch from Module 00)

H100 BF16, B* ~ 295 (weights only; lower with KV traffic):
  gamma=4 -> speculate while B < ~59
  gamma=1 -> speculate while B < ~147

In practice KV traffic lowers B*, so measure rather than assume.`,
        note: 'This is why the same feature can show 2.8x and 0.6x on the same hardware and model. Report the operating point or the number is meaningless.'
      }],
      pitfalls: [
        'A/B testing speculation at fixed concurrency and deploying to variable traffic.',
        'Forgetting that rejected drafts still consumed KV blocks that must be rolled back correctly. Bugs here manifest as rare corruption.',
        'Enabling speculation and chunked prefill without checking their interaction on mixed steps.',
        'Not exporting acceptance rate, then being unable to explain a throughput regression after a traffic shift.'
      ],
      build: [
        'Implement or configure batch-adaptive speculation. Measure goodput under a realistic diurnal load pattern with and without it.',
        'Instrument acceptance rate per request class (chat vs RAG vs code) and confirm they differ. Set per-class policy.'
      ],
      checks: [
        'Why must the scheduler know about speculation when computing its token budget?',
        'Derive the batch size above which gamma=4 speculation should be disabled on your hardware.',
        'Why does speculative decoding pair well with prefix caching but poorly with chunked prefill?'
      ],
      res: ['r-vllm-spec-blog', 'r-vllm-spec-docs', 'r-gpu-mode-l22', 'r-speculators', 'r-modular-spec']
    },

    {
      id: 'u-structured', tier: 3,
      title: 'Constrained decoding and structured output',
      summary: 'FSMs, grammars, and making JSON-mode fast enough to leave on.',
      why: 'A large fraction of production LLM traffic is not chat: it is function calling, structured extraction, and agent tool use, all of which require the output to conform to a schema. The naive implementation — generate, parse, retry on failure — burns tokens and adds tail latency. Constrained decoding instead masks invalid tokens at each step so that only conforming outputs are possible.\n\nThe engineering problem is that the mask must be computed per request per step over a 128k vocabulary, on CPU, without stalling the GPU. Outlines showed how to precompute an FSM so that the per-step cost is a table lookup; XGrammar extended it to context-free grammars with a persistent stack and a precomputed adaptive token mask, reaching near-zero overhead. Knowing how they work tells you when the overhead will bite.',
      core: [
        'Regex/JSON-schema constraints compile to a finite-state machine over the *token* vocabulary, not characters. The compilation is the expensive part and is cacheable per schema.',
        'At each step, the FSM state determines an allowed-token bitmask; disallowed logits are set to -inf before sampling.',
        'Context-free grammars need a pushdown automaton. XGrammar splits tokens into context-independent (precomputable) and context-dependent (checked at runtime) sets, which is what makes it fast.',
        'Mask computation is CPU work that can be overlapped with the GPU forward pass. If your stack does it synchronously, it lands directly on ITL.',
        'Constrained decoding changes the output distribution — it is renormalization over a subset, not the target distribution. That is intended, but it means quality comparisons must be like for like.',
        'Schema compilation caching is essential: compiling a complex JSON schema can take hundreds of milliseconds, which is fine once and fatal per request.',
        'Tokenizer edge cases (byte-level BPE, tokens spanning grammar boundaries) are where these systems get subtle. Trust the library; do not write your own.'
      ],
      math: [{
        title: 'Where the overhead lives',
        body: `Per step, per request:
  mask bits = vocab_size (e.g. 128k bits = 16 KB)
  batch 256 -> 4 MB of mask per step

FSM approach: state -> precomputed allowed set, O(1) lookup
CFG  approach: stack state -> context-independent cache hit,
               plus O(small) runtime check for dependent tokens

Budget: step time is 10-40 ms. Mask work for the whole batch
must fit inside that, overlapped with the GPU forward.
Unoverlapped, at 0.2 ms/request/step and batch 256:
  51 ms of CPU work per step -> now the bottleneck.`,
        note: 'This is why "structured output slows us down 30%" is usually an integration problem rather than an algorithmic one.'
      }],
      pitfalls: [
        'Recompiling the grammar per request. Cache by schema hash.',
        'Applying constraints to requests that do not need them.',
        'Comparing constrained and unconstrained output quality directly. The distributions differ by construction.',
        'Assuming a valid JSON schema guarantees semantically correct output. It guarantees syntax only.'
      ],
      build: [
        'Serve a JSON-schema-constrained endpoint. Measure ITL with and without constraints at batch 1 and batch 64. The batch-64 gap tells you whether your stack overlaps mask computation.',
        'Time schema compilation for a nested schema and verify caching works by issuing the same schema twice.'
      ],
      checks: [
        'Why must the FSM be over tokens rather than characters?',
        'What does XGrammar precompute, and why does that make CFG constraints practical?',
        'Why does constrained decoding not preserve the model\'s distribution, and when does that matter?'
      ],
      res: ['r-outlines', 'r-xgrammar', 'r-sglang-paper', 'r-vllm-structured-docs']
    }
  ]
},

/* ==================================================================== 07 */
{
  id: 'm07', num: '07',
  title: 'Distributed inference',
  tagline: 'Tensor, pipeline and expert parallelism, disaggregation, and multi-node topology.',
  focus: 'Once a model does not fit on one GPU — or fits but too slowly — you are doing distributed systems, and every parallelism strategy is a different trade between memory, latency, and communication. This module derives the communication volume of each strategy, explains why MoE changed the calculus, and covers prefill/decode disaggregation, which is the architectural expression of the asymmetry from Module 00.',
  outcome: 'A justified parallelism configuration for a specific model and SLO, with measured communication overhead, plus a working disaggregated prefill/decode deployment or a written analysis of why it is not worth it for you.',
  units: [
    {
      id: 'u-tensor-parallel', tier: 2,
      title: 'Tensor parallelism and what it costs',
      summary: 'Split every matrix, all-reduce twice per layer, and pay for it in latency.',
      why: 'Tensor parallelism is the default way to make a large model fit and to make decode faster, because it divides both the weight memory and the per-GPU bandwidth requirement by the TP degree. That last part is the real reason it helps latency: decode is bandwidth bound, and TP=8 gives you 8x the aggregate bandwidth.\n\nThe cost is two all-reduces per layer on the critical path. For an 80-layer model that is 160 synchronizations per token, and at a few tens of microseconds each the communication can become a third of your step time. This is why TP within an NVLink domain is fine and TP across nodes usually is not, and why the right TP degree is a measurement rather than a maximum.',
      core: [
        'Megatron-style TP: column-split the QKV and MLP up/gate projections, row-split the output and down projections. One all-reduce after attention, one after the MLP.',
        'Memory per GPU falls as `1/TP` for weights, activations and (usually) KV. Aggregate bandwidth rises as `TP`. Both help decode.',
        'Communication volume per all-reduce per GPU with ring: `2(P-1)/P × tokens × d_model × dtype_bytes`. It does *not* shrink with TP degree — more GPUs means more, smaller messages plus more latency.',
        'At decode, messages are tiny (batch × d_model) so the all-reduce is latency bound, not bandwidth bound. Small-message latency, not link bandwidth, is the limiter.',
        'KV heads constrain TP: with GQA and 8 KV heads, TP>8 must replicate KV heads, so KV memory stops dividing. This is a hard wall on many models.',
        'TP across nodes multiplies latency by an order of magnitude. Keep TP inside the NVLink domain; use PP or DP to cross nodes.',
        'Sequence/context parallelism splits along the sequence dimension for long contexts and is complementary to TP, mainly helping prefill and long-context memory.'
      ],
      math: [{
        title: 'Tensor parallel communication',
        body: `Per transformer layer, 2 all-reduces of the activation tensor:
    message = tokens_in_step * d_model * bytes

Ring all-reduce, bytes moved per GPU per all-reduce:
    2 * (TP-1)/TP * message

Total per step per GPU:
    C = L * 2 * 2 * (TP-1)/TP * tokens * d_model * bytes

70B: L=80, d=8192, BF16, decode batch 64, TP=8:
    message = 64 * 8192 * 2 = 1.05 MB
    per all-reduce per GPU = 2*(7/8)*1.05 MB = 1.84 MB
    per step = 80 * 2 * 1.84 MB = 294 MB
    at 450 GB/s NVLink = 0.65 ms

Compare against a ~15 ms decode step -> ~4%. Acceptable.
Now over PCIe at 50 GB/s: 5.9 ms -> 39%. Not acceptable.
And 160 separate small collectives have fixed latency
(~5-10 us each) = 0.8-1.6 ms regardless of size.`,
        note: 'The fixed-latency term is why decode TP scaling stops well before the bandwidth math says it should. Measure a TP sweep; do not assume linearity.'
      }],
      pitfalls: [
        'Setting TP greater than the number of KV heads on a GQA model and losing the KV memory benefit silently.',
        'Running TP across PCIe or across nodes for a latency-sensitive service.',
        'Assuming TP=8 gives 8x the decode throughput. Communication and fixed latency eat a growing share.',
        'Ignoring that TP requires all ranks to be healthy; a single slow GPU sets the pace for every step.'
      ],
      build: [
        'Sweep TP = 1, 2, 4, 8 for a model that fits at TP=1. Plot per-stream decode tokens/s and aggregate throughput. Find where scaling breaks and explain it with the formula above.',
        'Measure NCCL all-reduce latency for the message sizes your decode step uses (`nccl-tests` or an equivalent microbenchmark). Compare to your measured step-time overhead.'
      ],
      checks: [
        'Why does TP help decode latency at all, given it adds communication?',
        'What happens to KV memory at TP=16 with 8 KV heads?',
        'Why is decode all-reduce latency bound rather than bandwidth bound?'
      ],
      res: ['r-megatron', 'r-pope-scaling', 'r-modular-parallelism', 'r-ultrascale', 'r-gpu-mode-l17', 'r-ml-engineering']
    },

    {
      id: 'u-other-parallel', tier: 3,
      title: 'Pipeline, data, and context parallelism',
      summary: 'The other three axes, when each is right, and how they compose.',
      why: 'TP is not the only axis and is often not the best one. Pipeline parallelism splits layers across devices, communicates only activations at stage boundaries (tiny compared to all-reduces), and therefore crosses slow links well — at the cost of a pipeline bubble that hurts latency. Data parallelism replicates the model and is the right answer whenever the model fits, because it has zero inference-time communication. Context parallelism splits the sequence and is the answer for very long contexts.\n\nReal deployments compose these: DP across nodes, TP within a node, EP for experts, CP for long context. Knowing the communication signature of each is what lets you pick a configuration instead of copying one.',
      core: [
        'Pipeline parallelism: layers split into stages; only stage-boundary activations cross the link (`tokens × d_model × bytes` per boundary, once). Excellent for slow interconnects.',
        'PP\'s cost is the bubble. With continuous batching and enough in-flight microbatches the bubble can be largely filled, but per-request latency still traverses all stages sequentially.',
        'Data parallelism: full replicas, zero inference communication, linear throughput scaling, no latency benefit. Always prefer it when the model fits and you need throughput.',
        'Context/sequence parallelism: splits the sequence dimension so each device holds part of the KV. Ring Attention passes KV blocks around the ring, overlapping communication with computation. This is how 1M-token contexts are served.',
        'Composition rule of thumb: TP inside a node (fast links, latency sensitive), PP or DP across nodes (slow links), CP when a single request\'s KV does not fit, EP for MoE.',
        'Every added axis multiplies operational complexity: more failure modes, harder rollouts, more ways to misconfigure. Prefer the fewest axes that meet the SLO.',
        'For inference specifically, DP > TP > PP in preference order when you have the choice, because DP has no communication and PP has the worst latency profile.'
      ],
      tables: [{
        title: 'Parallelism axes for inference',
        cols: ['Axis', 'Splits', 'Comm per step', 'Helps latency?', 'Use when'],
        rows: [
          ['Data (DP)', 'Requests', 'None', 'No', 'Model fits; need throughput'],
          ['Tensor (TP)', 'Every matrix', '2 all-reduces per layer', 'Yes (bandwidth)', 'Model too big or decode too slow; fast links'],
          ['Pipeline (PP)', 'Layers', 'Activations at stage boundaries', 'No (adds bubble)', 'Crossing slow links; capacity bound'],
          ['Expert (EP)', 'MoE experts', 'All-to-all per MoE layer', 'Depends', 'Large MoE models'],
          ['Context (CP)', 'Sequence', 'KV blocks around a ring', 'Yes for long ctx', 'Single-request KV exceeds one GPU']
        ]
      }],
      pitfalls: [
        'Using PP for a latency-critical service and being surprised by the per-request latency floor.',
        'Combining TP and PP without checking that the layer count divides evenly; stage imbalance sets the pace.',
        'Choosing TP when DP would have worked, and paying communication for nothing.',
        'Forgetting that DP replicas each need their own KV cache and their own prefix cache — hence cache-aware routing.'
      ],
      build: [
        'For one model and one SLO, enumerate three feasible configurations across TP/PP/DP and compute memory per GPU, comm per step, and expected latency for each. Pick one and justify it in writing.',
        'If you have two nodes, measure the same model at TP=8 within a node versus TP=16 across nodes. The gap is the lesson.'
      ],
      checks: [
        'Why does pipeline parallelism cross slow links better than tensor parallelism?',
        'When does context parallelism become necessary rather than merely helpful?',
        'Why is data parallelism the preferred axis for inference whenever it is available?'
      ],
      res: ['r-ultrascale', 'r-pope-scaling', 'r-ring-attention', 'r-modular-parallelism', 'r-loongserve', 'r-ml-engineering']
    },

    {
      id: 'u-moe', tier: 3,
      title: 'Serving Mixture-of-Experts models',
      summary: 'Sparse compute, dense memory, all-to-all communication, and load imbalance.',
      why: 'MoE broke several assumptions at once. A model with 671B total and 37B active parameters has the FLOP profile of a 37B model and the memory profile of a 671B one, so your capacity planning must use two different numbers. Routing is data dependent, so expert load is imbalanced and varies per batch. Expert parallelism replaces all-reduce with all-to-all, which has completely different scaling behaviour.\n\nSince the frontier open models are now predominantly MoE, this is no longer a specialist topic. The practical questions are: how do you fit total parameters, how do you keep experts busy, and how do you stop one hot expert from setting the pace for the whole step.',
      core: [
        'Active vs total parameters: FLOPs and decode weight-read scale with active parameters *per token*, but all experts must be resident somewhere, so memory scales with total.',
        'Expert parallelism shards experts across GPUs. Each MoE layer needs an all-to-all dispatch (tokens to their experts) and an all-to-all combine (results back). Two all-to-alls per MoE layer.',
        'All-to-all volume depends on top-k routing and token count, and is sensitive to topology in a way all-reduce is not. NVLink domains matter enormously.',
        'Load imbalance: routing is learned and skewed. A step is as slow as the busiest expert. Mitigations include capacity factors, dropless routing with padding, and expert replication for hot experts.',
        'At decode with small batch, each expert sees very few tokens, so expert GEMMs are tiny and inefficient. MoE decode efficiency improves markedly with batch size — the opposite of the intuition that sparse means cheap.',
        'Grouped/fused MoE kernels (grouped GEMM, MegaBlocks-style block-sparse) exist to make the many-small-GEMM problem tractable.',
        'Large-scale MoE serving (DeepSeek-style) combines EP with DP and TP, uses dedicated communication kernels (DeepEP), and often disaggregates prefill and decode to different EP configurations.'
      ],
      math: [{
        title: 'MoE cost structure',
        body: `Per token:
  FLOPs   ~ 2 * (P_dense + k * P_expert)      k = top-k experts
  memory  ~ P_total * bytes                    (all experts resident)

DeepSeek-V3-class: 671B total, 37B active, k=8 of 256 routed
  weight memory  (FP8)  = 671 GB
  decode FLOPs/token    = 2 * 37e9 = 74 GFLOP

Bytes read per decode step at batch B, EP across G GPUs:
  each GPU reads only its resident experts that were selected.
  Expected experts touched per GPU grows with B and saturates:
    E[distinct experts] = N_e * (1 - (1 - k/N_e)^B)
  B=1, k=8, N_e=256 -> 8 experts read
  B=64            -> ~223 experts read   (nearly dense!)

-> MoE decode is weight-bandwidth-cheap only at small batch.
   At large batch you read almost every expert anyway.`,
        note: 'This is the single most counterintuitive fact about MoE serving, and it explains why MoE throughput curves look nothing like dense ones.'
      }],
      pitfalls: [
        'Sizing memory from active parameters. You need total.',
        'Assuming MoE always reduces decode cost. At high batch nearly all experts are touched and the sparsity benefit largely disappears.',
        'Ignoring expert load imbalance; a single hot expert can dominate step time.',
        'Running expert parallelism across a slow link. All-to-all is far less forgiving than all-reduce.'
      ],
      build: [
        'Compute the expected number of distinct experts touched per step for your MoE model at batch 1, 8, 64, 256. Plot it. Explain the throughput curve you then measure.',
        'Instrument per-expert token counts for a real batch and compute the imbalance factor (max/mean). If it exceeds 2, that is your bottleneck.'
      ],
      checks: [
        'Why does MoE decode efficiency behave non-monotonically with batch size?',
        'Why is all-to-all more topology sensitive than all-reduce?',
        'What are your options when one expert is consistently hot?'
      ],
      res: ['r-moe-hf', 'r-megablocks', 'r-deepseek-v3', 'r-deepep', 'r-fastmoe', 'r-modular-parallelism']
    },

    {
      id: 'u-disagg', tier: 3,
      title: 'Prefill/decode disaggregation',
      summary: 'Give each phase its own machine, transfer the KV, and stop the two from interfering.',
      why: 'Module 00 established that prefill and decode want opposite hardware and opposite scheduling. Chunked prefill is a compromise that lets them share a GPU peacefully. Disaggregation is the other answer: run them on separate pools, transfer the KV cache from the prefill node to the decode node, and let each pool be sized, batched and even quantized independently.\n\nDistServe and Splitwise showed the gains are real — large improvements in goodput under strict SLOs — and Mooncake showed how to build the KV transfer layer that makes it practical. It is also more machinery: a transfer path, a scheduler that spans pools, and a new failure mode when the transfer is slow. Know the break-even before you build it.',
      core: [
        'Prefill pool: compute-optimized, large batches of prompt tokens, high MFU. Decode pool: bandwidth- and capacity-optimized, many concurrent sequences.',
        'The KV for a completed prefill must reach the decode node before the first decode step. Transfer time = `kv_bytes_per_token × prompt_tokens / link_bandwidth`, and it lands directly in TTFT.',
        'Because the transfer is on the TTFT path, disaggregation needs a fast link — NVLink, or RDMA over a fast fabric. On slow links it is worse than colocation.',
        'The main win is SLO isolation: prefill spikes no longer stutter decode, so you can run each pool closer to its own knee. Reported goodput improvements are large under tight latency constraints.',
        'Ratio sizing: the prefill:decode node ratio follows from your input:output token ratio and each phase\'s throughput. It is a computation, not a guess, and it changes with traffic mix.',
        'Layer-wise transfer overlaps the copy with the remaining prefill layers, hiding most of the transfer latency. This is the key implementation trick.',
        'Modern stacks (NVIDIA Dynamo, llm-d, vLLM/SGLang disaggregated modes, Mooncake) implement this with pluggable KV connectors such as NIXL.'
      ],
      math: [{
        title: 'Sizing and break-even',
        body: `Node ratio:
  prefill nodes / decode nodes
    = (input_tokens/s / prefill_throughput_per_node)
    / (output_tokens/s / decode_throughput_per_node)

10:1 input:output at 10k output tok/s:
  input 100k tok/s ; prefill node does 25k tok/s -> 4 prefill nodes
  decode node does 2.5k tok/s -> 4 decode nodes
  ratio 1:1 (and it shifts with every traffic change)

Transfer cost added to TTFT:
  t_xfer = prompt_tokens * kv_bytes_per_token / BW

4k prompt, GQA-8 70B (320 KiB/token), 400 Gb/s RDMA (50 GB/s):
  t_xfer = 4096 * 327680 / 50e9 = 26.8 ms   -> acceptable
Same over 25 Gb/s (3.1 GB/s):
  t_xfer = 433 ms                            -> fatal

With layer-wise overlap, effective cost ~ t_xfer / L per layer,
mostly hidden behind remaining prefill compute.`,
        note: 'Compute this before building anything. Disaggregation is an architecture, not a flag, and the link speed decides whether it can work at all.'
      }],
      pitfalls: [
        'Building disaggregation before exhausting chunked prefill tuning and prefix caching. Most services do not need it.',
        'Sizing the pool ratio once. Traffic mix moves and the ratio must move with it, which means autoscaling both pools independently.',
        'Ignoring the transfer in the TTFT budget.',
        'Losing prefix cache benefits because the prefill pool\'s cache is not visible to the decode pool or to other prefill nodes.'
      ],
      build: [
        'Compute your transfer cost and pool ratio for your model, traffic mix and network. Decide whether disaggregation is viable and write down the number that decided it.',
        'If viable and you have the hardware, stand up a disaggregated deployment and compare goodput under a tight TTFT+TPOT SLO against the colocated baseline. Goodput, not throughput.'
      ],
      checks: [
        'Why does disaggregation improve goodput more than it improves raw throughput?',
        'How does layer-wise KV transfer hide most of the transfer latency?',
        'What network bandwidth do you need for disaggregation to beat colocation for your model?'
      ],
      res: ['r-distserve', 'r-splitwise', 'r-mooncake', 'r-modular-disagg', 'r-dynamo', 'r-nixl', 'r-llm-d']
    },

    {
      id: 'u-multinode', tier: 4,
      title: 'Multi-node topology, collectives, and failure',
      summary: 'NCCL, NVLink domains, RDMA, and what happens when one GPU is slow.',
      why: 'At single-node scale the interconnect is a constant you can ignore. At multi-node scale it is the design. Which collectives run over NVLink and which over the fabric, whether your NVLink domain is 8 GPUs or 72, and how the job behaves when one link degrades are the questions that decide whether a large deployment works.\n\nThis unit is deliberately at the frontier tier: most readers will not run multi-node inference, but everyone serving a large MoE eventually will, and the failure modes are unintuitive enough to be worth reading before you meet them.',
      core: [
        'Collectives: all-reduce (TP), all-to-all (EP), all-gather/reduce-scatter (sharded weights), point-to-point (PP stage boundaries, KV transfer). Each has a different topology sensitivity.',
        'NVLink domain size is the key architectural parameter. An 8-GPU node bounds TP at 8; NVL72-class systems make much larger NVLink domains possible and change what configurations are sensible.',
        'RDMA (InfiniBand or RoCE) provides tens of GB/s between nodes with low latency — orders below NVLink but far above TCP. GPUDirect avoids a host bounce.',
        'NVSHMEM and similar one-sided communication libraries allow finer-grained overlap than collective calls, which matters for MoE all-to-all and for KV transfer.',
        'Stragglers: any collective is as slow as its slowest participant. A thermally throttled GPU, an ECC-retired page, or a degraded link silently sets the pace for the entire deployment.',
        'Failure handling for inference is different from training: there is no checkpoint to resume from, so the practical answer is fast detection, drain, and replacement, with the router steering traffic away.',
        'Measure the fabric independently (nccl-tests, ib diagnostics) before blaming the engine. Most "the engine is slow at scale" reports are fabric problems.'
      ],
      pitfalls: [
        'Assuming NCCL picked the optimal algorithm and topology. It usually does; when it does not, the symptom is a mysterious 2x.',
        'Mixing GPU generations or link speeds in one TP group.',
        'Not monitoring per-GPU step time. Aggregate metrics hide a single straggler completely.',
        'Deploying multi-node TP for latency. It is almost always the wrong axis to cross nodes with.'
      ],
      build: [
        'Run nccl-tests for all-reduce and all-to-all at your actual message sizes, within a node and across nodes. Record bus bandwidth and latency; these are the constants for all your later estimates.',
        'Deliberately throttle one GPU (power limit) and observe the effect on aggregate throughput. Confirm your monitoring would have caught it.'
      ],
      checks: [
        'Why is a straggler more damaging to inference than to data-parallel training?',
        'When would you cross nodes with pipeline parallelism rather than tensor parallelism?',
        'What is the first measurement you take when multi-node throughput is below expectation?'
      ],
      res: ['r-gpu-mode-l17', 'r-gpu-mode-l67', 'r-nccl-docs', 'r-ml-engineering', 'r-ultrascale', 'r-deepep']
    }
  ]
},

/* ==================================================================== 08 */
{
  id: 'm08', num: '08',
  title: 'Production serving',
  tagline: 'Routing, autoscaling, multi-tenancy, reliability, and the parts that are not the model.',
  focus: 'Everything so far has been about one engine on one set of GPUs. A service is a fleet: a router in front, an autoscaler underneath, many tenants sharing it, and an on-call rotation behind it. This module covers the layer that turns an engine into a product, where the dominant failure modes are queueing, cache locality and capacity rather than kernels.',
  outcome: 'A deployed service with cache-aware routing, sane autoscaling on a queue-depth signal, per-tenant limits, and a dashboard whose alerts you would actually trust at 3am.',
  units: [
    {
      id: 'u-router', tier: 2,
      title: 'The router: load balancing that knows about KV cache',
      summary: 'Why round-robin is wrong, and what to route on instead.',
      why: 'A generic load balancer treats replicas as interchangeable. LLM replicas are not: each has a different prefix cache, a different queue, and a different set of sequences already in flight. Routing a request to the replica that already holds its prefix can eliminate the entire prefill. SGLang measured a 3.8x higher cache hit rate and 1.9x throughput from cache-aware routing alone — a bigger win than most kernel work.\n\nThe router is also where you implement priority, fairness, model multiplexing and overload shedding, because it is the only component that sees the whole fleet. Getting it right is high leverage and comparatively cheap.',
      core: [
        'Round-robin and least-connections both ignore the two things that matter: prefix cache locality and expected request cost.',
        'Cache-aware routing: hash prefix chunks and maintain an approximate map from prefix to replica. Route to the replica with the longest matching prefix, subject to a load cap so hot prefixes do not create hotspots.',
        'Least-outstanding-tokens beats least-connections, because requests are wildly heterogeneous in cost. A 200-token request and a 100k-token request are not one connection each.',
        'Queue depth at the engine, not CPU or GPU utilization, is the signal that correlates with latency. Export it and route on it.',
        'Session affinity for multi-turn chat is the single easiest cache win: send turn N+1 to the replica that served turn N.',
        'The router must shed load. When every replica is past its knee, queueing more requests only converts a throughput problem into a timeout problem. Reject early with a clear error.',
        'Production implementations: SGLang router, vLLM production stack, `llm-d` inference scheduler, NVIDIA Dynamo\'s KV-aware router, plus Envoy/Gateway API extensions for LLM traffic.'
      ],
      math: [{
        title: 'The value of cache-aware routing',
        body: `Request with shared prefix P and unique suffix U.

Random routing over R replicas:
  P(prefix already resident) ~ 1/R   (for a warm prefix)
  E[prefill tokens] = |P|*(1 - 1/R) + |U|

Cache-aware routing:
  E[prefill tokens] ~ |U|   (when the prefix is anywhere in the fleet)

System prompt 2000 tok, user turn 100 tok, R=8:
  random     : 2000*0.875 + 100 = 1850 prefill tokens
  cache-aware:                     100 prefill tokens
  -> 18x less prefill work

Guard against hotspots: cap replica load at
  load_i < (1 + eps) * mean_load
and fall back to least-outstanding-tokens when the cap binds.`,
        note: 'This is why routing belongs in a course on inference optimization rather than a course on infrastructure. It is often the largest single lever available.'
      }],
      pitfalls: [
        'Using a stock HTTP load balancer with default settings and wondering why the prefix cache hit rate is low.',
        'Cache-aware routing without a load cap, creating a hot replica for a popular system prompt.',
        'Routing on CPU utilization. It is nearly meaningless for a GPU-bound service.',
        'Health checks that hit a trivial endpoint. A replica can pass `/health` while its queue is 300 deep.'
      ],
      build: [
        'Deploy 2+ replicas behind a cache-aware router. Measure prefix cache hit rate and TTFT against round-robin on a workload with a shared system prompt.',
        'Add a load cap and verify with a deliberately skewed workload that no replica exceeds it.'
      ],
      checks: [
        'Why is least-outstanding-tokens a better signal than least-connections?',
        'How do you prevent cache-aware routing from creating hotspots?',
        'What health signal actually predicts latency for an LLM replica?'
      ],
      res: ['r-sglang-v04', 'r-modular-routing', 'r-preble', 'r-llm-d', 'r-dynamo', 'r-vllm-prod-stack']
    },

    {
      id: 'u-autoscale', tier: 3,
      title: 'Capacity, autoscaling, and cold starts',
      summary: 'Little\'s Law, the queueing knee, and the brutal arithmetic of a 3-minute model load.',
      why: 'Autoscaling an LLM service is harder than autoscaling a web service for one reason: the scale-up latency is enormous. Acquiring a GPU node, pulling a 140 GB image or checkpoint, loading weights and warming CUDA graphs can take minutes. By the time the new replica is ready, the traffic spike is over. This changes the entire design: you scale on a leading indicator, you keep warm capacity, and you accept a lower average utilization than you would for stateless services.\n\nUnderneath is the queueing theory from Module 00. Latency does not degrade linearly with load; it degrades hyperbolically near the knee. Knowing where your knee is, and staying below it, is capacity planning.',
      core: [
        'Little\'s Law: `L = lambda × W`. Concurrency in the system equals arrival rate times residence time. Every capacity question is a rearrangement of this.',
        'The knee: as utilization `rho` approaches 1, queueing delay scales like `rho/(1-rho)`. At 90% utilization queueing delay is 9x service time; at 95% it is 19x. Target 60-75%, not 95%.',
        'Scale on queue depth or on TTFT p95, not on GPU utilization. GPU utilization is ~100% whenever anything is running and tells you nothing.',
        'Cold start budget: node acquisition + image pull + weight load + CUDA graph capture + warmup. Measure each. Typical total is 2-10 minutes, and it dominates every scaling decision.',
        'Mitigations: pre-pulled images or a shared image cache, weight loading from local NVMe or a fast object store with parallel readers, model preloading into page cache, keeping a warm pool, and over-provisioning by the amount of traffic that arrives during a cold start.',
        'Scale-down must be graceful: stop admitting, drain in-flight sequences (which can take minutes for long generations), then terminate. Aggressive scale-down kills user requests.',
        'Cost of over-provisioning versus cost of a violated SLO is a business decision. Compute both explicitly rather than picking a utilization target by feel.'
      ],
      math: [{
        title: 'Capacity from Little\'s Law and cold-start headroom',
        body: `Concurrency needed:
  L = lambda * W
  100 req/s, avg 8 s residence -> L = 800 concurrent sequences

Replicas needed:
  R = ceil(L / L_max_per_replica)
  L_max from KV memory: KV_avail / kv_bytes_per_seq

Queueing headroom (M/M/1 intuition):
  W_queue = W_service * rho/(1-rho)
  rho=0.7 -> 2.3x ; rho=0.9 -> 9x ; rho=0.95 -> 19x

Cold-start headroom:
  spare_capacity >= d(lambda)/dt * T_coldstart
  Traffic ramps 20 req/s per minute, cold start 4 min
    -> keep 80 req/s of spare capacity warm.`,
        note: 'That last line is the number most teams never compute, and it is why their autoscaler always arrives late.'
      }],
      pitfalls: [
        'Autoscaling on GPU utilization. It saturates immediately and carries no load information.',
        'Targeting 90%+ utilization and being confused by tail latency.',
        'Scale-down that terminates pods with in-flight long generations.',
        'Ignoring that a new replica starts with a cold prefix cache and is therefore slower than steady-state for several minutes.'
      ],
      build: [
        'Measure your full cold start with a stopwatch, broken into phases. Then compute the spare capacity your traffic ramp requires.',
        'Load test to find the knee: sweep QPS and plot p95 TTFT. Mark the utilization at which p95 doubles. Set your autoscaler target below it.'
      ],
      checks: [
        'Why is GPU utilization a useless autoscaling signal for LLM serving?',
        'Compute the concurrency needed for 50 req/s with 12 s average residence time.',
        'Why is a freshly scaled-up replica slower than a warm one even after weights load?'
      ],
      res: ['r-littles-law', 'r-vllm-prod-stack', 'r-kserve', 'r-ray-serve', 'r-modal-examples', 'r-llm-d']
    },

    {
      id: 'u-multitenancy', tier: 3,
      title: 'Multi-tenancy, LoRA, and fairness',
      summary: 'Many models and many customers on shared GPUs without letting one ruin the rest.',
      why: 'Very few deployments serve one model to one customer. The economics push hard toward sharing: GPUs are expensive, and a single fine-tuned variant rarely saturates one. Multi-LoRA serving lets hundreds of adapters share one base model\'s weights, and multi-tenant scheduling lets many customers share one engine. Both introduce the same problem — isolation — and the same failure mode, which is one tenant\'s pathological request starving everyone else.\n\nFairness in an LLM scheduler is genuinely subtle because requests have unbounded, unknown cost. A request that generates 32k tokens occupies a slot for minutes. Classic fair queueing assumes you can preempt cheaply; here preemption means recomputing or swapping KV.',
      core: [
        'Multi-LoRA (S-LoRA/Punica style): keep base weights resident, swap small low-rank adapters per request, batch requests with different adapters together using segmented gather-matmul kernels. Hundreds of adapters on one GPU.',
        'Adapter memory is small (`2 × r × d × layers`), but adapter *switching* and unified paging of adapter weights are where the engineering is.',
        'Fair scheduling: virtual-time or deficit-round-robin over *tokens served*, not requests served, because token cost varies by orders of magnitude.',
        'Per-tenant limits worth enforcing: max concurrent requests, max tokens in flight, max output length, max context length, and a token-per-second rate limit. Enforce at the router.',
        'Priority classes with preemption: interactive traffic preempts batch traffic. Preemption cost is the KV recompute or swap, so preempt the request with the least KV invested when possible.',
        'Noisy-neighbour detection: track per-tenant token consumption and queue occupancy. The tenant sending 100k-token prompts at 10 QPS is visible in seconds if you look.',
        'Complete isolation requires separate replicas. Everything short of that is a trade, so decide deliberately which tenants get dedicated capacity.'
      ],
      math: [{
        title: 'LoRA memory and the sharing win',
        body: `LoRA adapter memory, rank r, per layer, for a target matrix d_in x d_out:
  2 * r * (d_in + d_out) * bytes   (A and B factors)

Llama-70B, r=16, adapters on q,k,v,o (d=8192), 80 layers, BF16:
  per layer = 4 matrices * 16 * (8192+8192) * 2 B = 2.1 MB
  total     = 80 * 2.1 MB = 168 MB per adapter

Base weights (FP8) = 70 GB.
On an 80 GB GPU: 70 GB base + KV + ~40 adapters resident.
Dedicated deployment: 40 adapters x 70 GB = 2.8 TB.
-> ~35x fewer GPUs.`,
        note: 'The economics of multi-LoRA are so strong that the interesting question is never whether to do it, only how to isolate tenants once you have.'
      }],
      pitfalls: [
        'Fair queueing over requests rather than tokens. One tenant issuing long generations then consumes almost everything while appearing fair.',
        'No max output length. A single runaway generation can hold a slot for the entire timeout window.',
        'Batching adapters naively by looping over them, which destroys batch efficiency. Use segmented kernels.',
        'Assuming per-tenant limits at the API gateway are enough. The engine also needs to know, or the queue is still shared.'
      ],
      build: [
        'Serve 5+ LoRA adapters on one base model. Measure throughput against a dedicated deployment per adapter and compute the GPU saving.',
        'Simulate a noisy neighbour (long prompts, long outputs, high QPS) and verify your limits protect the other tenants\' p95. If they do not, fix the scheduler, not the gateway.'
      ],
      checks: [
        'Why must fairness be measured in tokens rather than requests?',
        'What makes preemption expensive in an LLM scheduler, and how does that change the fair-queueing design?',
        'Compute LoRA adapter memory for rank 32 on a 8B model and decide how many fit alongside the base weights.'
      ],
      res: ['r-slora', 'r-punica', 'r-vllm-lora-docs', 'r-vtc-fairness', 'r-llumnix']
    },

    {
      id: 'u-reliability', tier: 3,
      title: 'Reliability, observability, and the on-call view',
      summary: 'What to export, what to alert on, and how LLM services actually fail.',
      why: 'LLM services fail in ways that monitoring built for web services will not catch. The request rate is fine, the error rate is zero, the GPUs are at 100% utilization — and every user is waiting 40 seconds for a first token because the KV cache is full and the queue is 400 deep. Or the model is returning fluent nonsense because a quantized checkpoint was rolled out with a bad scale. Neither shows up as an error.\n\nThe fix is to instrument the things that are specific to this workload: queue depth, KV utilization, preemption rate, cache hit rate, acceptance rate, and token-level latency percentiles. Then alert on the ones that lead rather than lag.',
      core: [
        'Golden signals for LLM serving: TTFT p50/p95/p99, ITL p95, end-to-end p95, queue depth, KV cache utilization, preemption/recompute rate, prefix cache hit rate, running vs waiting request counts, tokens/s in and out.',
        'Alert on leading indicators: queue depth and KV utilization rise before latency does. Alerting on latency alone means you always find out last.',
        'Preemption rate is the single best "you are over capacity" signal. A non-zero and rising preemption rate means requests are being recomputed.',
        'Streaming complicates error handling: a failure after the first token cannot be turned into an HTTP 500. Define what a partial response means and how clients should handle it.',
        'Timeouts must be per-phase. A single end-to-end timeout either kills legitimate long generations or lets stuck requests hold slots forever. Separate TTFT timeout, inter-token stall timeout, and max total duration.',
        'Quality regressions are silent. Ship a canary evaluation that runs a fixed prompt set against production and diffs against a reference; a quantization or config change can degrade output with no operational symptom.',
        'Rollouts: shadow traffic first, then canary with quality and latency gates, then progressive rollout. Weight loading times make rollbacks slow, so gate early.',
        'Common incidents worth rehearsing: KV exhaustion under a long-context spike, one straggler GPU, prefix cache thrash after a system prompt change, tokenizer mismatch after a model update, and a draft model outage in a speculative deployment.'
      ],
      tables: [{
        title: 'Symptom to cause map',
        cols: ['Symptom', 'Likely cause', 'First check'],
        rows: [
          ['TTFT up, ITL flat', 'Queueing at admission', 'Queue depth, arrival rate'],
          ['ITL up, TTFT flat', 'Batch too large or prefill interference', 'Batch size, chunked prefill config'],
          ['Both up, throughput down', 'KV exhaustion / preemption', 'KV utilization, preemption counter'],
          ['Throughput down after deploy', 'Config or precision change', 'Diff the engine args; check quant path'],
          ['One replica slow', 'Straggler GPU or cold cache', 'Per-GPU step time, cache hit rate'],
          ['Quality complaints, no metric change', 'Quantization or template regression', 'Canary eval, prompt template diff'],
          ['Periodic latency spikes', 'CUDA graph recapture / cache eviction', 'Trace a spike with the profiler']
        ]
      }],
      pitfalls: [
        'Averaging TTFT. The distribution is heavily skewed by prompt length; always use percentiles and segment by input length bucket.',
        'A single end-to-end timeout for all requests.',
        'No canary eval, so quality regressions are found by customers.',
        'Dashboards without queue depth or KV utilization, which is most default dashboards.'
      ],
      build: [
        'Build the dashboard: the eight golden signals above, on one screen, with percentiles. Then run a load test past the knee and confirm you can see the failure develop.',
        'Write the symptom-to-cause runbook for your own service and validate at least three rows by deliberately inducing the failure.'
      ],
      checks: [
        'Why does GPU utilization near 100% tell you nothing about health?',
        'Which metric rises first when you exceed capacity, and why?',
        'How should a streaming endpoint report an error that occurs after the first token?'
      ],
      res: ['r-vllm-metrics-docs', 'r-guidellm', 'r-sre-book', 'r-vllm-prod-stack', 'r-artificial-analysis']
    },

    {
      id: 'u-longcontext-prod', tier: 4,
      title: 'Long context in production',
      summary: 'What changes at 128k and beyond, and why the cost model stops being linear.',
      why: 'Long context breaks the assumptions of every earlier module simultaneously. Prefill cost grows quadratically in attention, KV memory grows linearly and quickly dominates weights, a single request can exceed one GPU\'s memory, prefix caching becomes both more valuable and harder, and the scheduler\'s fairness model collapses because one request can consume an entire replica.\n\nProduction long-context serving is therefore a distinct discipline: chunked prefill becomes mandatory, context parallelism or KV offload becomes necessary, sparse attention and KV compression stop being research and become operational, and the pricing model usually has to change because cost is no longer linear in tokens.',
      core: [
        'Cost structure at long context: prefill attention is `O(S^2)`, prefill FFN is `O(S)`, KV memory is `O(S)`, decode attention per step is `O(S)`. Past roughly 32k, attention starts to dominate prefill.',
        'A 1M-token context on a 70B GQA-8 model needs roughly 320 GB of KV at FP16 — more than any single GPU. Context parallelism, KV offload or aggressive compression is not optional.',
        'Decode also slows down: every step reads the whole KV cache, so ITL grows linearly with context length. Users notice this directly.',
        'Sparse and selective attention at decode (Quest, MInference, DuoAttention) exploit the fact that only a small fraction of KV entries matter per step. These give real speedups at long context with modest quality cost.',
        'Prefix caching is enormously valuable here — a cached 100k-token document turns a 30-second prefill into nothing — but the cache is huge, so tiered storage (GPU / CPU / NVMe) becomes necessary.',
        'Scheduling: a single 1M-token request must be chunked, and while it prefills it will starve other requests unless the token budget explicitly bounds its share per step.',
        'Pricing: if cost is superlinear in context but you charge linearly, long-context users are subsidized by everyone else. Either price by a cost model or cap context.',
        'Quality at long context is not guaranteed by support for it. Needle-in-a-haystack and RULER-style evaluations frequently show effective context far below advertised context.'
      ],
      math: [{
        title: 'Where long context hurts',
        body: `Prefill FLOPs, model params N, sequence S, layers L, d_model d:
  FFN + proj : 2 * N * S
  attention  : 2 * L * S^2 * d       (scores + values)

70B, L=80, d=8192:
  S=4k   : 2*70e9*4096 = 5.7e14 ; attn = 2*80*4096^2*8192 = 2.2e13   (4%)
  S=128k : 2*70e9*131072 = 1.8e16 ; attn = 2*80*131072^2*8192 = 2.3e16 (56%)
  S=1M   : proj 1.4e17 ; attn 1.4e18                                   (91%)

KV memory, GQA-8, head_dim 128, FP16:
  per token = 2 * 80 * 8 * 128 * 2 B = 327,680 B = 320 KiB
  S=128k -> 40 GB ; S=1M -> 320 GB

Decode step KV read at S=1M = 320 GB
  at 3.35 TB/s -> 95 ms per step for ONE request.`,
        note: 'That last number is the whole story: at 1M context, a single decode step spends 95 ms reading KV before touching the weights. Long-context ITL is a KV bandwidth problem.'
      }],
      pitfalls: [
        'Advertising a context length you have not evaluated for quality at that length.',
        'Charging linearly per token when your cost is quadratic in prefill.',
        'Letting one long request monopolize a replica because the token budget was not enforced.',
        'Enabling sparse attention globally without measuring quality on long-context retrieval tasks specifically.'
      ],
      build: [
        'Measure TTFT and ITL at 1k, 8k, 32k, 128k input tokens. Plot both. Fit the TTFT curve and confirm the quadratic term appears where the math says it should.',
        'Compute your true cost per request at each length and compare against your pricing. Decide whether to cap, tier, or reprice.'
      ],
      checks: [
        'At what context length does attention overtake the projections in prefill cost for your model?',
        'Why does ITL grow with context length even though the model weights read per step is constant?',
        'What does an effective-context evaluation measure that a context-length spec does not?'
      ],
      res: ['r-ring-attention', 'r-minference', 'r-quest', 'r-duoattention', 'r-needle-haystack', 'r-ruler', 'r-loongserve', 'r-lmcache']
    }
  ]
},

/* ==================================================================== 09 */
{
  id: 'm09', num: '09',
  title: 'Economics and the frontier',
  tagline: 'Unit costs, the reasoning-model shift, hardware alternatives, and how to keep learning.',
  focus: 'The last module closes the loop from Module 00: every technique in this course exists to move a number that is ultimately measured in dollars per million tokens at a given SLO. It also covers what is changing right now — test-time compute, agentic traffic, alternative silicon, and post-attention architectures — and how to keep the whole picture current after you finish.',
  outcome: 'A defensible unit-economics model for your service, a written architecture decision record, and a published artefact that demonstrates the whole chain from roofline to dollars.',
  units: [
    {
      id: 'u-unit-economics', tier: 2,
      title: 'Unit economics: dollars per million tokens',
      summary: 'Turning throughput into cost, and cost into an architecture decision.',
      why: 'Every optimization in this course is ultimately justified in one currency. A 30% throughput improvement is worth exactly 30% of your GPU bill, which for a serious deployment can be more than an engineer\'s salary — and this is why unit economics belongs at the end, once you can actually predict the throughput.\n\nThe model itself is simple arithmetic, but the inputs are where people go wrong: they use peak throughput instead of throughput at their SLO, they forget utilization, and they ignore that input and output tokens have wildly different costs. Getting the model right lets you compare a quantization project against a hardware upgrade against a routing change on the same axis.',
      core: [
        'Base identity: `cost per token = GPU $/hour / (tokens per second × 3600)`, then adjust for utilization.',
        'Use throughput at your SLO, not peak throughput. The difference is frequently 2-3x and it is where most cost models become fiction.',
        'Input and output tokens cost differently. Input tokens are prefill (compute bound, fast, batchable); output tokens are decode (bandwidth bound, slow). A 10:1 input:output workload has a completely different cost structure from a 1:10 one, and this is why every API prices them separately.',
        'Utilization is the multiplier everyone forgets. At 40% average utilization your effective cost per token is 2.5x the number your benchmark suggested.',
        'Prefix caching directly removes cost: cached input tokens cost nothing to prefill, which is why providers offer discounted cached-input pricing.',
        'Compare against API pricing honestly. Self-hosting wins above a volume threshold that you can compute; below it, it does not, and knowing the threshold is the point.',
        'Cost per *request* or per *task* is often the more meaningful business metric than cost per token, particularly for agentic traffic where one user action becomes many calls.'
      ],
      math: [{
        title: 'The full cost model',
        body: `Given:
  P     = GPU price per hour (per node)
  T_out = output tokens/s per node at your SLO
  T_in  = input  tokens/s per node at your SLO
  U     = average utilization (0-1)
  r     = input:output token ratio in your traffic

Blended cost per 1M output tokens:
  C_out = P / (T_out * 3600 / 1e6) / U
Cost per 1M input tokens:
  C_in  = P / (T_in  * 3600 / 1e6) / U

Worked: 8xH100 node at $24/hr, 70B FP8,
  T_out = 2,800 tok/s at TTFT p95 < 1s, ITL p95 < 50ms
  T_in  = 42,000 tok/s
  U     = 0.55

  C_out = 24 / (2800*3600/1e6) / 0.55 = 24 / 10.08 / 0.55 = $4.33 /1M
  C_in  = 24 / (42000*3600/1e6) / 0.55 = 24 / 151.2 / 0.55 = $0.29 /1M

Blended at r = 5:1 input:output:
  per 1M output tokens you also serve 5M input tokens
  total = 4.33 + 5*0.29 = $5.78 per 1M output tokens

Break-even vs an API at $A per 1M output (same blend):
  self-host wins when your monthly volume V satisfies
    V * (A - 5.78)/1e6 > engineering + ops cost per month`,
        note: 'Run this with your own numbers before starting any optimization project. It tells you which of the nine preceding modules is worth your time.'
      }],
      pitfalls: [
        'Using peak benchmark throughput. Use throughput at the SLO you actually promise.',
        'Forgetting utilization, which is the single largest error in most cost models.',
            'Pricing input and output identically.',
        'Comparing self-hosted cost against API price without including engineering, on-call and idle capacity.'
      ],
      build: [
        'Build the cost model for your service in a spreadsheet or the calculator here. Include utilization and the input:output ratio from real traffic.',
        'Compute the cost delta for each optimization you have made in this course. Rank them by dollars saved. The ranking will surprise you.'
      ],
      checks: [
        'Why do input and output tokens have different costs, and by roughly what factor for your model?',
        'Compute cost per million output tokens for your setup at 40% and 80% utilization.',
        'At what monthly volume does self-hosting beat an API for your workload?'
      ],
      res: ['r-tensor-economics', 'r-llm-numbers', 'r-artificial-analysis', 'r-databricks-perf', 'r-modular-benchmarks']
    },

    {
      id: 'u-testtime', tier: 3,
      title: 'Reasoning models and test-time compute',
      summary: 'When output length explodes by 10-100x, every trade-off you tuned moves.',
      why: 'Reasoning models changed the workload shape more than any optimization changed the system. A model that emits 5,000 thinking tokens before a 200-token answer converts a prefill-heavy workload into an overwhelmingly decode-heavy one, and decode is the expensive phase. Everything you tuned for a 500-token output — batch size, KV budget, speculative gamma, disaggregation ratio — is now mistuned.\n\nAgentic traffic compounds this: one user action becomes a chain of calls with growing context, each re-prefilling a conversation that got longer. The combination is why prefix caching and KV reuse moved from optimization to necessity.',
      core: [
        'Output length distribution becomes long-tailed and much heavier. Median and p99 output length can differ by 50x, which destroys any scheduling assumption based on average length.',
        'KV per request grows throughout generation, so a batch that fit at admission may not fit 3,000 tokens later. This is exactly the case preemption exists for, and it fires far more often.',
        'Decode-heavy traffic makes bandwidth and KV capacity the binding constraints. Weight quantization and KV quantization become the highest-value optimizations.',
        'Speculative decoding often works *better* on reasoning traces than on chat, because thinking text is more predictable. Measure acceptance separately for thinking and answer phases.',
        'Agentic loops re-send growing conversations. Without prefix caching, cost grows quadratically in the number of turns; with it, linearly. This is the single largest lever for agent traffic.',
        'Budget forcing and early stopping (cap thinking tokens, stop on confidence) are product-level controls with direct cost impact — and they belong in the serving layer, not only the prompt.',
        'Batch/offline inference for non-interactive agent work is dramatically cheaper because you can run at large batch without an ITL constraint. Route it separately.'
      ],
      math: [{
        title: 'Why agent chains need prefix caching',
        body: `Agent loop, N turns, each turn adds t tokens of context,
generation g tokens per turn.

Without prefix caching, turn i re-prefills i*t tokens:
  total prefill = t * N(N+1)/2      -> O(N^2)

With prefix caching, turn i prefills only the new t tokens:
  total prefill = t * N              -> O(N)

N=20 turns, t=500:
  no cache : 105,000 prefill tokens
  cache    :  10,000 prefill tokens
  -> 10.5x less prefill work, and it grows with N.

Reasoning output cost shift:
  chat      : 500 in / 300 out  -> decode-dominated but modest
  reasoning : 500 in / 8000 out -> decode is ~96% of the work
  decode tok/s is the ONLY number that matters here.`,
        note: 'If you serve agents or reasoning models and have not enabled prefix caching, stop reading and go do that first.'
      }],
      pitfalls: [
        'Capacity planning on average output length when the distribution is long-tailed.',
        'Keeping a max-tokens default tuned for chat, which truncates reasoning mid-thought.',
        'Serving agentic and interactive traffic from the same pool with the same SLO. They want different configurations.',
        'Not measuring cache hit rate on agent traffic, where it should be very high and is the main cost control.'
      ],
      build: [
        'Plot the output-length distribution of your real traffic on a log scale. Compute median, p90, p99. Re-derive your KV budget from p90 rather than the mean.',
        'Measure the end-to-end cost of a 20-turn agent conversation with and without prefix caching. Report the ratio.'
      ],
      checks: [
        'Why does agent traffic cost grow quadratically without prefix caching?',
        'Which optimizations rise in value when output length increases 10x, and which fall?',
        'Why might speculative decoding perform better on reasoning traces than on chat?'
      ],
      res: ['r-s1-testtime', 'r-testtime-scaling', 'r-deepseek-r1', 'r-sglang-blog', 'r-vllm-prefix-docs', 'r-modular-batch-offline']
    },

    {
      id: 'u-hardware-frontier', tier: 4,
      title: 'Hardware alternatives and the silicon landscape',
      summary: 'TPUs, Trainium, MI300X, Groq, Cerebras, and how to evaluate a chip honestly.',
      why: 'NVIDIA is the default, not the only option, and the alternatives fail or succeed for structural reasons you can reason about from Module 00. A chip with enormous SRAM and no HBM (Cerebras, Groq) attacks the decode bandwidth wall directly and can deliver extraordinary tokens per second per stream — at the cost of needing many chips to hold a model. A chip with more HBM per dollar (MI300X) wins on capacity-bound workloads. A chip with a mature compiler and poor kernel coverage will benchmark well on the vendor\'s model and badly on yours.\n\nThe skill is evaluating a chip against *your* workload\'s position on the roofline, and knowing that the software ecosystem is usually the binding constraint rather than the silicon.',
      core: [
        'The decode wall is a bandwidth wall, so any architecture that raises effective bandwidth per parameter read wins on latency. That is the entire thesis behind SRAM-based inference chips.',
        'Groq LPU / Cerebras: keep weights in on-chip SRAM with bandwidth orders of magnitude above HBM. Result is very high single-stream tokens/s. Constraint is aggregate SRAM, so a large model spans many chips and the economics depend on utilization.',
        'Google TPU: strong systolic-array compute, good interconnect (ICI), mature for large-scale serving via JAX/XLA. Different programming model; not a drop-in.',
        'AWS Inferentia/Trainium: cost-optimized, requires the Neuron SDK and compilation; kernel coverage for the newest model architectures lags.',
        'AMD MI300X: 192 GB HBM3 at 5.3 TB/s is genuinely competitive hardware; ROCm coverage in vLLM/SGLang has improved substantially but remains the thing to verify, not assume.',
        'Evaluation checklist: does your model architecture have kernel coverage, what quantization formats are supported, what is the measured (not spec) bandwidth, how does the interconnect behave for your parallelism plan, and what is the actual migration cost.',
        'Never compare spec sheets. Run your own benchmark from Module 04, at your own SLO, with your own model. Vendor benchmarks are correct and irrelevant.'
      ],
      pitfalls: [
        'Comparing peak TFLOPS across vendors, especially when one number includes sparsity.',
        'Assuming a supported framework means supported kernels for your specific attention variant.',
        'Ignoring the compilation step in the deployment loop for TPU/Neuron; it changes your release process.',
        'Evaluating single-stream latency for a throughput-bound service, or vice versa.'
      ],
      build: [
        'Take one alternative accelerator you have access to, run your Module 04 benchmark suite unchanged, and produce a like-for-like cost-per-million-tokens comparison.',
        'Write the evaluation checklist result for that chip: kernel coverage, formats, measured bandwidth, migration cost. Recommend or reject in writing.'
      ],
      checks: [
        'Why does an SRAM-based architecture help decode latency so dramatically, and what does it cost?',
        'What is the first thing you verify when evaluating a non-NVIDIA accelerator?',
        'Why are peak-TFLOPS comparisons across vendors nearly meaningless for inference?'
      ],
      res: ['r-modal-gpu-glossary', 'r-groq-lpu', 'r-tpu-book', 'r-mi300x-vllm', 'r-mlperf-inference', 'r-artificial-analysis']
    },

    {
      id: 'u-frontier', tier: 4,
      title: 'The moving frontier: architectures, kernels, and what to watch',
      summary: 'Linear attention, hybrids, diffusion LLMs, and how to read the field going forward.',
      why: 'The techniques in this course are stable enough to teach, but the frontier moves every few months, and the useful skill is not knowing today\'s state of the art — it is being able to read a new paper and immediately place it on the roofline. Almost every new inference technique is doing one of four things: reducing bytes moved, reducing FLOPs, increasing arithmetic intensity, or trading quality for one of the above.\n\nThis unit surveys the live directions and, more importantly, gives you the classification habit that makes new work easy to evaluate.',
      core: [
        'The four-question test for any new technique: what does it reduce (bytes/FLOPs/latency/memory), what does it cost (quality/complexity/memory), at what batch size does it help, and does it compose with what I already run?',
        'Linear and sub-quadratic attention (Mamba/SSM, RWKV, gated linear attention): constant-size recurrent state instead of a growing KV cache. Removes the decode bandwidth wall for context length entirely; quality on recall-heavy tasks is the open question.',
        'Hybrid architectures (a few full-attention layers among many linear ones) are the pragmatic middle and are shipping in production models. They change KV math substantially — recompute your memory model per layer type.',
        'Attention variants continue to consolidate: MLA (DeepSeek) compresses KV into a latent, and its serving characteristics differ enough from GQA that engines need specific kernels.',
        'Diffusion and any-order language models offer parallel token generation, breaking the sequential dependency that makes decode bandwidth bound. Early, but structurally interesting.',
        'Kernel-level frontier: FlashAttention-4 and Blackwell-specific kernels, ThunderKittens-style tile abstractions, and increasingly compiler-generated kernels. The gap between hand-written and generated is closing.',
        'Where to watch: arXiv cs.LG/cs.DC, MLSys and OSDI proceedings, vLLM/SGLang release notes and RFCs, GPU MODE lectures, and the engineering blogs of serving companies. Release notes are underrated — they tell you what actually shipped.',
        'Keep your benchmark harness from Module 04. It is the tool that converts every new claim into a number for your workload, and it is the most durable thing you will build in this course.'
      ],
      pitfalls: [
        'Chasing new techniques before exhausting the boring ones. Prefix caching, chunked prefill and FP8 beat most novel methods and are already in your engine.',
        'Evaluating a new method at a batch size you do not run.',
        'Assuming a paper\'s speedup composes with your existing optimizations. Speedups multiply only when they attack different bottlenecks.',
        'Reading only papers. Release notes and issue trackers carry more operational signal.'
      ],
      build: [
        'Pick one paper published in the last three months. Classify it with the four-question test, predict its speedup on your workload from first principles, then measure it. Compare prediction to reality.',
        'Write down the three techniques you have not yet tried, ranked by predicted dollars saved, with the arithmetic that produced the ranking.'
      ],
      checks: [
        'Apply the four-question test to linear attention. What does it reduce and what does it cost?',
        'Why do two independent 2x speedups rarely produce a 4x end-to-end speedup?',
        'What is your process for evaluating a new inference paper?'
      ],
      res: ['r-mamba', 'r-rwkv', 'r-deepseek-v2', 'r-flash-linear-attn', 'r-thunderkittens', 'r-survey-inference', 'r-gpu-mode-lectures', 'r-awesome-llm-inference']
    },

    {
      id: 'u-capstone', tier: 3,
      title: 'Capstone: the end-to-end artefact',
      summary: 'One deployment, one report, one defensible set of numbers.',
      why: 'Everything in this course composes into a single deliverable: a served model whose performance you can predict from first principles, measure reproducibly, optimize deliberately, and price honestly. Producing that artefact is what converts the reading into competence, and it is also the single most effective thing you can show someone who is evaluating whether you know this material.\n\nThe brief below is deliberately demanding. It requires you to have done the work of every module, and it requires the numbers to agree with each other.',
      core: [
        'Pick one model and one workload with a real input/output length distribution. Do not use a synthetic uniform distribution.',
        'Predict, before measuring: memory breakdown, maximum concurrency from KV, critical batch size, TTFT and ITL at three concurrency levels, and cost per million tokens. Write the predictions down first.',
        'Measure with a reproducible harness: fixed seeds, warmup, controlled arrival process, percentiles, and error bars. Report your methodology precisely enough that someone else could reproduce it.',
        'Explain every gap between prediction and measurement. The gaps are where the real learning is, and an unexplained 2x is a bug in your understanding.',
        'Optimize in ranked order of predicted value: prefix caching, chunked prefill tuning, quantization, KV quantization, speculative decoding, parallelism, routing. Measure after each, one change at a time.',
        'Produce the final artefact: an architecture decision record with the configuration, the measured performance envelope, the cost model, the rejected alternatives with reasons, and the runbook.',
        'Publish it. The write-up is the proof of work, and explaining the reasoning to someone else is the last step that consolidates it.'
      ],
      build: [
        'Write the prediction document before touching a GPU. Seal it.',
        'Run the full benchmark sweep and produce the measured envelope. Diff against predictions and explain every discrepancy above 20%.',
        'Apply optimizations one at a time, re-measuring after each. Produce a waterfall chart of throughput and cost.',
        'Write the ADR and the runbook. Publish the whole thing with the harness code.'
      ],
      checks: [
        'Can you predict your service\'s throughput at a new batch size without measuring, and be within 20%?',
        'Can you explain every number in your cost model from hardware specs and model dimensions?',
        'Could someone else reproduce your benchmark from your write-up alone?'
      ],
      res: ['r-ttft-repo', 'r-modular-handbook', 'r-guidellm', 'r-ml-engineering', 'r-vllm-docs']
    }
  ]
}

];
