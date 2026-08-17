/* window.REFERENCE — formula sheet, hardware constants, model configs,
   benchmarking traps, glossary, and the presets the calculators read. */
(function () {
  var hardwarePresets = [
    { name: 'H100 SXM 80GB', mem: 80, bw: 3350, bf16: 989, fp8: 1979 },
    { name: 'H100 PCIe 80GB', mem: 80, bw: 2000, bf16: 756, fp8: 1513 },
    { name: 'H200 SXM 141GB', mem: 141, bw: 4800, bf16: 989, fp8: 1979 },
    { name: 'B200 192GB', mem: 192, bw: 8000, bf16: 2250, fp8: 4500, fp4: 9000 },
    { name: 'A100 SXM 80GB', mem: 80, bw: 2039, bf16: 312 },
    { name: 'A100 SXM 40GB', mem: 40, bw: 1555, bf16: 312 },
    { name: 'L40S 48GB', mem: 48, bw: 864, bf16: 181, fp8: 362 },
    { name: 'L4 24GB', mem: 24, bw: 300, bf16: 121, fp8: 242 },
    { name: 'A10G 24GB', mem: 24, bw: 600, bf16: 62.5 },
    { name: 'RTX 4090 24GB', mem: 24, bw: 1008, bf16: 165, fp8: 330 },
    { name: 'MI300X 192GB', mem: 192, bw: 5300, bf16: 1307, fp8: 2615 },
    { name: 'MI325X 256GB', mem: 256, bw: 6000, bf16: 1307, fp8: 2615 }
  ];

  var modelPresets = [
    { name: 'Llama 3.2 1B', layers: 16, dmodel: 2048, kvHeads: 8, headDim: 64, params: 1.24 },
    { name: 'Llama 3.2 3B', layers: 28, dmodel: 3072, kvHeads: 8, headDim: 128, params: 3.21 },
    { name: 'Llama 3.1 8B', layers: 32, dmodel: 4096, kvHeads: 8, headDim: 128, params: 8.03 },
    { name: 'Llama 3.1 70B', layers: 80, dmodel: 8192, kvHeads: 8, headDim: 128, params: 70.6 },
    { name: 'Llama 3.1 405B', layers: 126, dmodel: 16384, kvHeads: 8, headDim: 128, params: 406 },
    { name: 'Mistral 7B v0.3', layers: 32, dmodel: 4096, kvHeads: 8, headDim: 128, params: 7.24 },
    { name: 'Mixtral 8x7B (46.7B total)', layers: 32, dmodel: 4096, kvHeads: 8, headDim: 128, params: 46.7 },
    { name: 'Qwen2.5 7B', layers: 28, dmodel: 3584, kvHeads: 4, headDim: 128, params: 7.62 },
    { name: 'Qwen2.5 32B', layers: 64, dmodel: 5120, kvHeads: 8, headDim: 128, params: 32.8 },
    { name: 'Qwen2.5 72B', layers: 80, dmodel: 8192, kvHeads: 8, headDim: 128, params: 72.7 },
    { name: 'Gemma 2 9B', layers: 42, dmodel: 3584, kvHeads: 8, headDim: 256, params: 9.24 },
    { name: 'Phi-3 mini 3.8B', layers: 32, dmodel: 3072, kvHeads: 32, headDim: 96, params: 3.82 },
    { name: 'DeepSeek-V3 (MLA latent 576)', layers: 61, dmodel: 7168, kvHeads: 1, headDim: 288, params: 671 },
    { name: 'GPT-2 style MHA 7B (worst case)', layers: 32, dmodel: 4096, kvHeads: 32, headDim: 128, params: 6.7 }
  ];

  window.REFERENCE = {
    hardwarePresets: hardwarePresets,
    modelPresets: modelPresets,
    formulas: [
      {
        title: 'Cost of a forward pass',
        blurb: 'Everything else in this track is a consequence of these four lines. Memorise them.',
        items: [
          { title: 'FLOPs',
            body: 'decode, one token   : F = 2 * N_params            (+ attention term)\nprefill, S tokens   : F = 2 * N_params * S\nattention (prefill) : F = 4 * B * S^2 * d_model * L\nbackward (training) : F = 4 * N_params   -> 6 N total, for contrast',
            note: 'The factor 2 is one multiply plus one add per parameter. The attention term is quadratic in `S` and independent of `N_params`; the two terms are equal at `S = N_params / (2 * L * d_model)`, which is about 30k tokens for Llama 3.1 8B and about 54k for Llama 3.1 70B.' },
          { title: 'Bytes moved per decode step',
            body: 'bytes = N_params * w_bytes                      (weights)\n      + 2 * L * n_kv * d_head * S_ctx * B * kv_bytes (KV cache)\n      + activations (small; usually ignorable)',
            note: 'At batch 1 the weight term dominates by orders of magnitude. At large batch or long context the KV term takes over — the crossover point is the single most useful thing to compute about your deployment.' },
          { title: 'KV cache size',
            body: 'bytes/token/req = 2 * L * n_kv_heads * d_head * kv_bytes\ntotal           = bytes/token/req * S_ctx * B\n\nMHA  : n_kv_heads = n_heads\nGQA  : n_kv_heads = n_heads / group_size\nMQA  : n_kv_heads = 1\nMLA  : replace (n_kv_heads * d_head) with the latent width',
            note: 'The leading 2 is key plus value. Llama 3.1 8B: 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KiB per token. A 128k-token context is 16 GiB — twice the weights.' },
          { title: 'Arithmetic intensity and the ridge point',
            body: 'AI = FLOPs / bytes_moved            [FLOP/byte]\nAI(decode, batch B)  = 2B / w_bytes\nAI(prefill, S tok)   = 2S / w_bytes\n\nridge point = peak_FLOPS / peak_bandwidth\nH100 SXM BF16 : 989e12 / 3.35e12 = 295 FLOP/byte\n\nAI < ridge  -> memory bound  -> optimise bytes\nAI > ridge  -> compute bound -> optimise FLOPs',
            note: 'Decode at batch 1 with BF16 weights has AI = 1. The ridge point is 295. You are 295x away from the compute roof, which is why decode throughput scales almost linearly with batch size until the KV cache runs out of room.' }
        ]
      },
      {
        title: 'Latency and throughput',
        items: [
          { title: 'The four metrics',
            body: 'TTFT  = queue_wait + prefill_time            (time to first token)\nTPOT  = mean inter-token latency after the first\nITL   = the per-token latency distribution (report p50/p95/p99)\nE2E   = TTFT + (N_out - 1) * TPOT\n\noutput throughput = generated tokens / wall second\ntotal  throughput = (prompt + generated) tokens / wall second',
            note: 'Never report a single throughput number without saying which of the last two you mean; they differ by the prefill:decode token ratio, often 10x.' },
          { title: 'Memory-bound decode time',
            body: 't_step = bytes_moved / (HBM_bw * efficiency)\ntok/s (one seq)   = 1 / t_step\ntok/s (aggregate) = B / t_step\n\nefficiency is 0.7-0.9 in practice, not 1.0',
            note: 'Llama 3.1 8B BF16 on one H100: 16.06 GB / 3.35 TB/s = 4.8 ms, so about 208 tok/s at batch 1 with a perfect kernel; expect 150-180 tok/s in reality.' },
          { title: 'Compute-bound prefill time',
            body: 't_prefill = (2 * N_params * S) / (peak_FLOPS * MFU)\n\nMFU for prefill is typically 0.4-0.6',
            note: 'Prefill is the one place in inference where you can genuinely saturate the tensor cores. If your measured prefill MFU is under 0.3, look for a kernel or a chunk size problem before you look anywhere else.' },
          { title: "Little's Law and capacity",
            body: 'L = lambda * W\n  L      = average concurrency in the system\n  lambda = arrival rate (requests/second)\n  W      = average time in system (seconds)\n\nmax sustainable lambda = L_max / W_target\nutilisation rho = lambda / mu   (must stay < 1)\nM/M/1 wait: W_q = rho / (mu * (1 - rho))',
            note: 'The queueing term explodes as rho approaches 1. Plan for rho around 0.6-0.7 if you have a p99 SLO; the last 30% of utilisation costs you an unbounded tail.' }
        ]
      },
      {
        title: 'Batching, scheduling, caching',
        items: [
          { title: 'Critical batch size',
            body: 'B_crit = ridge_point * w_bytes / 2\n\nH100 + BF16 weights : 295 * 2 / 2 = 295\nH100 + FP8 weights  : 295 * 1 / 2 = 148\nB200 + FP4 weights  : (2250/8) * 0.5 / 2 = 70',
            note: 'Below `B_crit` you are buying throughput almost for free by batching. Above it, extra batch buys you nothing but latency. Quantising the weights *lowers* the critical batch size, which is a subtle and frequently missed consequence.' },
          { title: 'Maximum concurrent requests',
            body: 'KV_budget = gpu_mem * util - weights - activations - fragmentation\nB_max     = KV_budget / (kv_bytes_per_token * S_ctx)\n\nwith paging, fragmentation is ~4%\nwithout paging, reserve-max wastes 60-80%',
            note: 'This, not the compute roof, is what actually caps concurrency on almost every deployment.' },
          { title: 'Prefix cache economics',
            body: 'saved_flops   = 2 * N_params * S_shared * hit_rate\neffective TTFT = (1 - h) * t_prefill(S) + h * t_prefill(S - S_shared)\n                 + cache_lookup\n\nblocks are hashed; a hit requires an exact\nblock-aligned prefix match from token 0',
            note: 'One differing byte at the start of the system prompt destroys the entire hit rate. Put the variable parts of the prompt last.' },
          { title: 'Chunked prefill token budget',
            body: 'per-iteration budget T = max_num_batched_tokens\nprefill chunk = T - (tokens claimed by running decodes)\n\nsmaller T -> lower TPOT jitter, higher TTFT\nlarger  T -> better prefill MFU, worse decode stalls',
            note: 'Sarathi-Serve calls the failure mode a generation stall: a long prefill monopolises an iteration and every decoding sequence pauses. The budget is the knob that trades TTFT against TPOT directly.' },
          { title: 'Offload versus recompute',
            body: 'recompute cost = 2 * N_params * S / (peak_FLOPS * MFU)\nfetch cost     = kv_bytes_per_token * S / link_bandwidth\n\nfetch if: kv_bytes/token / link_bw < 2*N_params/(FLOPS*MFU)',
            note: 'PCIe Gen5 x16 is about 63 GB/s each way. For Llama 3.1 8B the fetch is roughly 2 us/token against a recompute of about 0.03 ms/token, so fetching wins by a wide margin — until the link is contended.' }
        ]
      },
      {
        title: 'Speculative decoding',
        items: [
          { title: 'Expected accepted tokens',
            body: 'With draft length gamma and acceptance rate alpha:\n\nE[tokens per verify] = (1 - alpha^(gamma+1)) / (1 - alpha)\n\nc = cost ratio (draft step time / target step time)\nspeedup = E[tokens] / (gamma * c + 1)',
            note: 'The `+1` is the single target forward pass that verifies the whole draft. The break-even acceptance rate solves `E[tokens] = gamma*c + 1`; below it, speculation makes you slower.' },
          { title: 'Optimal draft length',
            body: 'maximise f(gamma) = (1 - a^(gamma+1)) / ((1 - a)(gamma*c + 1))\n\nrule of thumb, c ~ 0.1:\n  alpha 0.6 -> gamma 3-4\n  alpha 0.8 -> gamma 5-7\n  alpha 0.9 -> gamma 8-10',
            note: 'Gains saturate because `E[tokens]` is bounded by `1/(1-alpha)` however long the draft is, while the cost keeps growing linearly.' },
          { title: 'Why speculation dies at high batch',
            body: 'verify FLOPs = 2 * N_params * (gamma+1) * B\n\nAt small B the GPU has idle FLOPs, so verification is free.\nAt B > B_crit the system is already compute bound and\nthe extra (gamma+1)x FLOPs are paid in full.',
            note: 'This is why every engine gates speculation on batch size or queue depth. Speculation converts spare compute into latency; if there is no spare compute, there is nothing to convert.' }
        ]
      },
      {
        title: 'Parallelism and communication',
        items: [
          { title: 'Tensor parallel communication',
            body: 'all-reduce volume per layer = 2 * B * S * d_model * dtype_bytes\nring all-reduce time = 2*(P-1)/P * (bytes / link_bw)\n\ntwo all-reduces per transformer block\n(one after attention, one after MLP)',
            note: 'NVLink 4 gives about 450 GB/s per direction per GPU on H100; PCIe gives about 63 GB/s. That 7x gap is why TP above 2 across PCIe is usually a mistake.' },
          { title: 'Pipeline bubble',
            body: 'bubble fraction = (P - 1) / (M + P - 1)\n  P = pipeline stages, M = microbatches\n\nP=4, M=1  -> 0.75 wasted\nP=4, M=16 -> 0.16 wasted',
            note: 'In decode each step produces one token, so there is little to microbatch; pipeline parallelism is far more attractive for prefill and offline batch than for interactive decode.' },
          { title: 'What each axis costs you',
            body: 'TP : splits weights AND KV; per-layer all-reduce; needs NVLink\nPP : splits layers; point-to-point only; bubble; same KV per stage\nDP : full replica per GPU; no comms; no memory relief\nEP : MoE experts across GPUs; all-to-all; load imbalance risk\nSP : splits the sequence; ring/all-to-all; for long context',
            note: 'Rule of thumb: TP inside a node, PP or DP across nodes, EP only for MoE, SP only when one sequence will not fit.' }
        ]
      },
      {
        title: 'Economics',
        items: [
          { title: 'Cost per million tokens',
            body: 'cost_per_1M = ($/hr * 1e6) / (tok_per_s * 3600 * utilisation)\n\nblended, when input and output are priced differently:\n  cost = (P_in * n_in + P_out * n_out) / 1e6',
            note: 'Utilisation is where most cost models lie to themselves. A GPU that is busy 40% of the day costs 2.5x per token what the benchmark suggested.' },
          { title: 'Quantisation payback',
            body: 'memory saved = N_params * (w_bytes_old - w_bytes_new)\nnew B_max    = (KV_budget + memory_saved) / (kv/token * S_ctx)\nspeedup at batch 1 ~ w_bytes_old / w_bytes_new\nspeedup at large batch ~ 1 (compute bound)',
            note: 'Quantisation helps the memory-bound regime most. If you are already compute bound, W4A16 buys you capacity, not speed — and W4A4 is the only thing that buys speed.' },
          { title: 'Prefill:decode cost split',
            body: 'prefill FLOPs = 2 * N * n_in\ndecode FLOPs  = 2 * N * n_out\ndecode bytes  = n_out * (N * w_bytes + KV traffic)\n\nSummarisation (n_in >> n_out) : prefill dominated\nReasoning    (n_out >> n_in)  : decode dominated',
            note: 'These two workloads want different hardware, different parallelism and different batch sizes. Mixing them on one pool is the most common cause of an unexplainable p99.' }
        ]
      }
    ],
    hardware: {
      title: 'Accelerator constants worth memorising',
      cols: ['Accelerator', 'Memory', 'HBM BW', 'BF16 dense', 'FP8 dense', 'Ops:byte', 'Interconnect'],
      rows: [
        ['**H100 SXM**', '80 GB HBM3', '3.35 TB/s', '989 TFLOP/s', '1979 TFLOP/s', '**295**', 'NVLink 4, 900 GB/s bidir'],
        ['H100 PCIe', '80 GB HBM2e', '2.0 TB/s', '756 TFLOP/s', '1513 TFLOP/s', '378', 'PCIe 5.0, 128 GB/s bidir'],
        ['**H200 SXM**', '141 GB HBM3e', '4.8 TB/s', '989 TFLOP/s', '1979 TFLOP/s', '206', 'NVLink 4, 900 GB/s bidir'],
        ['**B200**', '192 GB HBM3e', '8.0 TB/s', '2250 TFLOP/s', '4500 TFLOP/s', '281', 'NVLink 5, 1.8 TB/s bidir'],
        ['A100 SXM 80GB', '80 GB HBM2e', '2.04 TB/s', '312 TFLOP/s', 'none', '153', 'NVLink 3, 600 GB/s bidir'],
        ['A100 SXM 40GB', '40 GB HBM2', '1.56 TB/s', '312 TFLOP/s', 'none', '200', 'NVLink 3, 600 GB/s bidir'],
        ['L40S', '48 GB GDDR6', '864 GB/s', '181 TFLOP/s', '362 TFLOP/s', '210', 'PCIe 4.0, no NVLink'],
        ['L4', '24 GB GDDR6', '300 GB/s', '121 TFLOP/s', '242 TFLOP/s', '403', 'PCIe 4.0 x16'],
        ['A10G', '24 GB GDDR6', '600 GB/s', '62.5 TFLOP/s', 'none', '104', 'PCIe 4.0'],
        ['RTX 4090', '24 GB GDDR6X', '1.01 TB/s', '165 TFLOP/s', '330 TFLOP/s', '164', 'PCIe 4.0, no NVLink'],
        ['**MI300X**', '192 GB HBM3', '5.3 TB/s', '1307 TFLOP/s', '2615 TFLOP/s', '247', 'Infinity Fabric, 896 GB/s'],
        ['MI325X', '256 GB HBM3e', '6.0 TB/s', '1307 TFLOP/s', '2615 TFLOP/s', '218', 'Infinity Fabric'],
        ['TPU v5e', '16 GB HBM2', '819 GB/s', '197 TFLOP/s (BF16)', 'n/a', '241', 'ICI mesh'],
        ['TPU v6e (Trillium)', '32 GB HBM', '1.64 TB/s', '918 TFLOP/s (BF16)', 'n/a', '560', 'ICI mesh']
      ],
      note: 'All TFLOP/s figures are **dense**. Vendor marketing usually quotes the 2:1 structured-sparsity number, which is exactly double and which no LLM inference kernel achieves. If a number you read is twice one of these, that is why. H100 L2 cache is 50 MB; SM count is 132 on SXM. Sources: NVIDIA H100/H200/A100/L40S/L4 datasheets, NVIDIA Blackwell architecture brief, AMD Instinct MI300X/MI325X datasheets, Google Cloud TPU v5e/v6e documentation.'
    },
    models: {
      title: 'Model configurations and their KV footprint',
      cols: ['Model', 'Layers', 'd_model', 'Heads (Q/KV)', 'd_head', 'KV bytes/token FP16', '128k context KV'],
      rows: [
        ['Llama 3.2 1B', '16', '2048', '32 / 8', '64', '32 KiB', '4.0 GiB'],
        ['Llama 3.2 3B', '28', '3072', '24 / 8', '128', '112 KiB', '14.0 GiB'],
        ['**Llama 3.1 8B**', '32', '4096', '32 / 8', '128', '**128 KiB**', '**16.0 GiB**'],
        ['**Llama 3.1 70B**', '80', '8192', '64 / 8', '128', '320 KiB', '40.0 GiB'],
        ['Llama 3.1 405B', '126', '16384', '128 / 8', '128', '504 KiB', '63.0 GiB'],
        ['Mistral 7B v0.3', '32', '4096', '32 / 8', '128', '128 KiB', '16.0 GiB'],
        ['Mixtral 8x7B', '32', '4096', '32 / 8', '128', '128 KiB', '16.0 GiB'],
        ['Qwen2.5 7B', '28', '3584', '28 / 4', '128', '56 KiB', '7.0 GiB'],
        ['Qwen2.5 72B', '80', '8192', '64 / 8', '128', '320 KiB', '40.0 GiB'],
        ['Gemma 2 9B', '42', '3584', '16 / 8', '256', '336 KiB', '42.0 GiB'],
        ['Phi-3 mini 3.8B (MHA)', '32', '3072', '32 / 32', '96', '384 KiB', '48.0 GiB'],
        ['**DeepSeek-V3 (MLA)**', '61', '7168', '128 / latent 576', '&mdash;', '**68.6 KiB**', '8.6 GiB'],
        ['Hypothetical 7B with MHA', '32', '4096', '32 / 32', '128', '512 KiB', '64.0 GiB']
      ],
      note: 'KV bytes/token = `2 * layers * kv_heads * head_dim * 2`. Compare the last row with Llama 3.1 8B: GQA with group size 4 is a straight 4x reduction, and it is the single highest-leverage architectural decision for serving cost. DeepSeek-V3 caches one 576-wide latent per layer instead of 128 heads, which is why a 671B model has a smaller KV cache than a 9B one.'
    },
    pitfalls: [
      { title: 'Reporting a mean latency', body: 'The ITL distribution under continuous batching is multimodal: a step that shares an iteration with a chunked prefill is several times slower than one that does not. A mean hides exactly the behaviour your users complain about. Report p50, p95 and p99, and plot the histogram at least once.' },
      { title: 'Benchmarking with a fixed concurrency instead of a fixed arrival rate', body: 'A closed-loop test with N workers never queues, because a new request only arrives when an old one finishes. Real traffic is open-loop and does queue. Closed-loop benchmarks systematically understate p99 and completely hide the capacity cliff. Use rate-based load generation.' },
      { title: 'Not warming up', body: 'The first requests pay CUDA context creation, kernel autotuning, CUDA graph capture, and torch.compile. On vLLM V1 that is several seconds. Discard the first 30-60 seconds or you are benchmarking the compiler.' },
      { title: 'Leaving the prefix cache on and not saying so', body: 'If every synthetic request shares a prompt prefix, you are measuring your cache, not your engine. Either randomise the prefix or disable prefix caching, and state which you did. A 40x TTFT improvement that vanishes in production is almost always this.' },
      { title: 'Ignoring the input and output length distribution', body: 'Fixed 512-in / 128-out is not a workload. Real traffic has a heavy-tailed input distribution and an output length that correlates with the input. Throughput at fixed lengths can differ by 5x from throughput on a real trace. Use ShareGPT, a sampled production trace, or at minimum a lognormal.' },
      { title: 'Comparing engines at different batch sizes', body: 'Engine A at `max_num_seqs=256` against engine B at its default of 64 is not a comparison. Fix the token budget, the max sequence count, the dtype, the KV dtype, the context length, and the parallelism, then compare. Publish the full config or the number means nothing.' },
      { title: 'Not measuring quality after every optimisation', body: 'Quantisation, KV eviction, sparse attention and aggressive sampling all trade accuracy for speed, and the trade is often invisible on perplexity. Run lm-evaluation-harness, plus a long-context probe such as RULER if you touched the KV cache. A 2x speedup that costs 4 points on GSM8K is a regression.' },
      { title: 'Trusting perplexity as a quality gate', body: 'Perplexity is dominated by the easy tokens. INT4 schemes routinely show a 0.1 perplexity delta and a 5-point drop on reasoning benchmarks. Perplexity is a smoke test, not a gate.' },
      { title: 'Measuring TTFT without separating queue time', body: 'TTFT = queue wait + prefill. If your TTFT regressed, the fix for a queueing problem (more replicas, better admission) and the fix for a prefill problem (chunking, better kernels, tensor parallelism) are completely different. vLLM exposes `vllm:time_in_queue_requests` separately; use it.' },
      { title: 'Using vendor TFLOP/s numbers with sparsity', body: 'Every marketing figure with an asterisk is the 2:1 structured-sparsity number and is exactly double the dense one. No LLM inference kernel achieves it. Halving it is not pessimism, it is correctness.' },
      { title: 'Assuming 100% of HBM bandwidth', body: 'A well-written kernel achieves 70-90% of peak. If your predicted decode latency is 30% better than measured, you probably assumed 1.0. Calibrate with a memory-bandwidth microbenchmark on the actual machine once, then reuse that number.' },
      { title: 'Forgetting that gpu_memory_utilization is not free memory', body: 'The KV budget is `total * util - weights - activations - CUDA context - NCCL buffers`. Setting `gpu_memory_utilization=0.95` on a multi-GPU deployment is a reliable way to OOM at the worst possible moment, because the NCCL buffers are allocated outside the accounted pool.' },
      { title: 'Benchmarking a single request and calling it throughput', body: 'Batch-1 latency and aggregate throughput are optimised by opposite decisions. A configuration that wins on one usually loses on the other. Always report the latency-throughput Pareto curve, not a point.' },
      { title: 'Letting the client be the bottleneck', body: 'A single-threaded Python client with per-token JSON parsing can cap out well below the server. Check client CPU usage. If the server GPU is at 40% and the numbers look bad, suspect the harness first.' },
      { title: 'Running the benchmark on a shared or throttled GPU', body: 'MIG partitions, other tenants, thermal throttling and a lower-than-expected clock all silently change results. Log `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` throughout the run.' },
      { title: 'Ignoring tokenizer differences when comparing models', body: 'Tokens per second is not comparable across models with different tokenizers. A model with a 30% more efficient tokenizer needs 30% fewer tokens for the same text. Compare characters or words per second when the models differ.' },
      { title: 'Reporting total token throughput as if it were output throughput', body: 'A summarisation workload with 4000-token inputs and 100-token outputs has a total throughput 41x its output throughput. Both are legitimate numbers; conflating them is not.' },
      { title: 'Assuming speculative decoding always helps', body: 'It helps at low batch and hurts at high batch, because it trades spare FLOPs for latency and at high batch there are none. Benchmark it at your actual production concurrency, not at batch 1.' },
      { title: 'Chasing MFU during decode', body: 'Decode is memory bound by two orders of magnitude. Its MFU will be 1-5% and that is correct, not a bug. Use bandwidth utilisation (MBU) as the decode efficiency metric and reserve MFU for prefill.' },
      { title: 'Not checking whether you are overhead bound', body: 'If GPU utilisation is high but kernel time in the trace is low, the bottleneck is the Python scheduler, kernel launches, or the sampler — not the model. CUDA graphs and a scheduler rewrite are the fix; a faster kernel is not. This was the entire motivation for vLLM V1.' },
      { title: 'Evaluating a KV compression method only on needle-in-a-haystack', body: 'Needle retrieval is nearly saturated and almost every eviction scheme passes it. RULER, multi-hop tracing and aggregation tasks are where they fail. Do not ship a KV budget you have only tested on needles.' },
      { title: 'Forgetting the attention FLOPs at long context', body: 'The `2N` per token approximation ignores a term that is quadratic in sequence length. At 32k+ context the attention term is a large fraction of prefill cost and your estimate will be badly wrong.' },
      { title: 'Tuning on a benchmark whose prompts all have the same length', body: 'Uniform lengths make continuous batching look far better than it is, because there is no ragged tail and no preemption. Length variance is what stresses the scheduler.' },
      { title: 'Not accounting for preemption', body: 'When the KV cache fills, the scheduler evicts or recomputes. Both show up as latency spikes in the tail and neither appears in an average. Watch `vllm:num_preemptions_total`; if it is nonzero under normal load, your concurrency limit is set wrong.' },
      { title: 'Believing a cost-per-token number without a utilisation figure', body: 'Cost per million tokens at 100% utilisation is a physics number. Cost per million tokens at your actual duty cycle is a business number. They differ by 2-5x and only one of them shows up on the invoice.' }
    ],
    glossary: [
      { term: 'Arithmetic intensity', def: 'FLOPs performed per byte moved from memory. The x-axis of the roofline. Decode at batch 1 sits near 1; the H100 ridge point is 295.' },
      { term: 'Ridge point', def: 'peak FLOP/s divided by peak bandwidth. The arithmetic intensity at which a kernel stops being memory bound and starts being compute bound.' },
      { term: 'Prefill', def: 'The single parallel forward pass over the whole prompt that populates the KV cache. Compute bound, high MFU.' },
      { term: 'Decode', def: 'The autoregressive loop that emits one token per step per sequence. Memory bound, low MFU, and where almost all of the wall-clock time goes.' },
      { term: 'TTFT', def: 'Time to first token: queue wait plus prefill. The metric users perceive as responsiveness.' },
      { term: 'TPOT / ITL', def: 'Time per output token, or inter-token latency. The metric users perceive as reading speed. Report its distribution, not its mean.' },
      { term: 'Goodput', def: 'Throughput counting only requests that met their SLO. The honest capacity metric; introduced in this context by DistServe.' },
      { term: 'MFU / MBU', def: 'Model FLOPs utilisation and model bandwidth utilisation: achieved fraction of the compute roof and the memory roof respectively. Use MFU for prefill, MBU for decode.' },
      { term: 'Continuous batching', def: 'Iteration-level scheduling: finished sequences leave the batch and new ones join at every step, instead of the batch being fixed for its whole lifetime. From Orca.' },
      { term: 'Chunked prefill', def: 'Splitting a long prefill across several iterations so it can share a token budget with running decodes, eliminating generation stalls. From Sarathi-Serve.' },
      { term: 'Generation stall', def: 'The pause every decoding sequence experiences when a long prefill monopolises an iteration.' },
      { term: 'PagedAttention', def: 'Storing the KV cache in fixed-size non-contiguous blocks with a per-sequence block table, exactly like OS virtual memory. Cuts fragmentation from 60-80% to under 4%.' },
      { term: 'Prefix caching', def: 'Reusing the KV blocks of a shared prompt prefix across requests. Requires an exact block-aligned match from token zero.' },
      { term: 'RadixAttention', def: 'A radix tree over cached prefixes with LRU eviction, allowing partial and branching prefix reuse rather than exact whole-prefix matching. From SGLang.' },
      { term: 'Attention sink', def: 'The first few tokens of a sequence, which attract disproportionate attention mass regardless of content. Evicting them destroys quality; keeping them permits an otherwise sliding window.' },
      { term: 'GQA', def: 'Grouped-query attention: several query heads share one KV head, cutting KV cache size by the group factor with minimal quality cost.' },
      { term: 'MLA', def: 'Multi-head latent attention: cache a single low-rank latent per layer and reconstruct K and V on the fly. DeepSeek-V2/V3. Around a 93% KV reduction.' },
      { term: 'Speculative decoding', def: 'Draft several tokens cheaply, verify them in one target forward pass, accept a prefix using modified rejection sampling. Output distribution is provably unchanged.' },
      { term: 'Acceptance rate (alpha)', def: 'The probability that a drafted token survives verification. Determines the entire speedup; below break-even, speculation is a slowdown.' },
      { term: 'Tensor parallelism', def: 'Splitting individual weight matrices across GPUs. Column-then-row partitioning gives exactly one all-reduce per transformer block. Needs fast interconnect.' },
      { term: 'Pipeline parallelism', def: 'Splitting layers across GPUs. Cheap point-to-point communication but pays a bubble, which decode cannot easily amortise.' },
      { term: 'Expert parallelism', def: 'Distributing MoE experts across GPUs, with an all-to-all dispatch and combine. Sensitive to routing imbalance.' },
      { term: 'PD disaggregation', def: 'Running prefill and decode on separate hardware pools so each can be parallelised and scaled independently. DistServe, Splitwise, Mooncake.' },
      { term: 'Outlier features', def: 'A small number of activation channels with magnitudes 20-100x the rest, which appear above roughly 6.7B parameters and break naive INT8 quantisation.' },
      { term: 'W4A16 / W8A8 / W4A4', def: 'Weight and activation precision notation. W4A16 saves memory and helps at low batch; W8A8 and W4A4 also cut compute and therefore help at high batch.' },
      { term: 'Calibration', def: 'Running a small representative dataset through the model to collect activation statistics used to choose quantisation scales. Its distribution matters more than its size.' },
      { term: 'CUDA graph', def: 'A captured, replayable sequence of kernel launches. Removes per-launch CPU overhead, which is decisive when steps take only a few milliseconds.' },
      { term: 'Overhead bound', def: 'The third regime alongside compute bound and memory bound: the GPU is idle waiting on the host. Diagnosed only with a timeline profiler.' },
      { term: 'Little\u2019s Law', def: 'L = lambda * W. Concurrency equals arrival rate times time in system. The bridge between an SLO and a replica count.' },
      { term: 'Preemption', def: 'Evicting or recomputing a running sequence when the KV cache fills. Invisible in averages, very visible in p99.' },
      { term: 'Structured output', def: 'Constrained decoding against a grammar or JSON schema by masking invalid tokens at each step. XGrammar and Outlines make the mask nearly free.' },
      { term: 'Continuous versus static batching', def: 'Static batching waits for the slowest sequence in the batch to finish before starting anything new; continuous batching does not. The difference is often 10-20x throughput.' }
    ]
  };
})();
