# Target diagnostics

No target-screen arm has been trained. Six arms: CLIP ViT-B/16 text,
EmbeddingGemma-300M, and Qwen3-Embedding-0.6B, each with InfoNCE and cosine.
Query encoder is CLIP ViT-B/16 text for every arm. Step budget is not frozen.
`run_target_screen` requires a frozen budget, clean manifests, and a
visual-cache sentinel. The sentinel was not created. EmbeddingGemma and
Qwen weights were not downloaded.

CLIP text: native 512, EOT, context 77.
EmbeddingGemma: native 768, mean pool, MRL leading 512 then renorm.
Prompts: `task: search result | query: `, `title: none | text: `,
`task: question answering | query: `.
Qwen3-Embedding-0.6B: native 1024, last token, MRL leading 512 then renorm.
Queries use `Instruct: {task}\nQuery: {text}`. Documents stay raw.

Descriptors, not objectives: effective rank, mean unit-vector norm,
paraphrase margin, linear probe, small nonlinear probe.

If text distinction is poor and the visual probe is good, change the target.
If both are good and conditional prediction is poor, change the supervision.
If visual probes are weak, change layer, view, resolution, or backbone
before adding a loss.

Clean-set visual cache estimate: 31.9255 GiB.
Five 512-d fp16 captions per image: 0.5402 GiB per encoder.
`FEATURE_DROPOUT_MASKS_BEFORE_ENCODER` is false.
Synthetic plumbing uses scope `synthetic_harness_not_a_result`.
