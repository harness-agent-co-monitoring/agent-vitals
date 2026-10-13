# Command line

Generated from the argparse help by `scripts/gen_cli_reference.py`; do not edit.

```text
usage: vitals [-h] [--version] COMMAND ...

agent-vitals: section-level read maps, monitoring and decision explanations for LLM agent harnesses.

Get episodes into a trace store with `vitals import` (logged calls) or `vitals serve` (live), recover attention
with `vitals replay`, then read the store with `vitals report` and `vitals dashboard`, and monitor it with
`vitals monitor` (checks, no model pass) or `vitals pipeline` (ledger, probes, explanations, webhook).

positional arguments:
  COMMAND
    import       import OpenAI-style chat logs (JSONL) into a trace store, grouped into episodes, ready for
                 replay and monitoring
    serve        OpenAI-compatible server that records every call into a trace store, with the /v1/vitals
                 sidecar
    replay       teacher-forced replay of a trace store: recover attention and next-token signals offline
                 from the logged tokens
    report       read map of a trace store: per turn and section (declared or automatic), read lift, peak
                 lift with its hottest line, consult ratio and contrast
    dashboard    local dashboard over the trace stores and runs found under ROOT: read maps, store overview,
                 drift monitor, pipeline alarms, decision explanations and Meta-Harness evolution
    annotate     declare sections for a static prompt file (Markdown headings, paragraphs or inline
                 markers), or check a declaration against a prompt or a stored episode
    monitor      checks over a trace store's turns with no labels and no model pass: low-confidence tool
                 arguments, values found nowhere in the context, section drift against a reference store
                 (alarms go to a cache, not into the store)
    ledger       section ledger of a store: at each turn's decision, per-head reading of every section, the
                 residual at a few layers, each section's mean residual and writes (teacher-forced passes;
                 written to a cache, not into the store)
    probe        behavior probes on the ledger: train one from labeled decisions, score a store's decisions
                 with the section measures (probe lift, z against the normal)
    pipeline     monitor a trace store end to end: replay, ledger, checks, probes, an explanation per flag
                 (read map + probe split), flags for the dashboard and an optional webhook; --watch repeats
                 it for live stores
    explain      explain decisions of stored turns: which sections pushed the action onset, by reading,
                 calibrated reading and exact ablation (results go to a cache, not into the store)
    sensitivity  compare output distributions after a controlled section replacement
    sections     compare section reading in failed and successful episodes (needs manifest meta.reward), and
                 label sections by their relation to the answer
    gate         check that flash attention on and off produce the same greedy tokens (the diagnostic
                 attention path)
    calibrate    per-model head census: retrieval heads, induction heads, position bias (one-time per model;
                 not the explain --weights file)
    engine       check the llama.cpp engine (vitals engine doctor)
    metaharness  Meta-Harness add-on: ingestion, replay drivers, digests, section tables and read maps
                 (vitals metaharness --help)

options:
  -h, --help     show this help message and exit
  --version      show program's version number and exit
```

## vitals import

```text
usage: vitals import [-h] --log LOG --store STORE --gguf GGUF [--prefix PREFIX] [--lib LIB]

options:
  -h, --help       show this help message and exit
  --log LOG        JSONL: {request, response} or {messages, tools, response} per line
  --store STORE    trace store directory to write
  --gguf GGUF      reference model GGUF (its tokenizer and chat template render the calls; no forward pass)
  --prefix PREFIX  episode id prefix when calls carry no session id
  --lib LIB        libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
```

## vitals serve

```text
usage: vitals serve [-h] --gguf GGUF [--host HOST] [--port PORT] [--store STORE] [--n-ctx N_CTX]
                    [--n-gpu-layers N_GPU_LAYERS] [--lib LIB] [--diagnostic-attention]
                    [--capture-layers CAPTURE_LAYERS] [--store-messages] [--capture-attention]
                    [--top-logprobs K] [--prompt-scores] [--spans-file SPANS_FILE] [--no-auto-sections]
                    [--reuse-kv]

options:
  -h, --help            show this help message and exit
  --gguf GGUF           path to the GGUF model to serve
  --host HOST           address to listen on (default 127.0.0.1)
  --port PORT           port (default 8033)
  --store STORE         trace store directory (default runs/traces)
  --n-ctx N_CTX         context window in tokens
  --n-gpu-layers N_GPU_LAYERS
                        layers offloaded to the GPU (default: all)
  --lib LIB             libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
  --diagnostic-attention
                        disable flash attention to enable per-head attention capture (slower)
  --capture-layers CAPTURE_LAYERS
                        comma-separated layer indices whose residual activations are saved to
                        activations.zarr, e.g. 3,15,31
  --store-messages      store request messages, tools and chat template options per turn (needed for replay;
                        may hold sensitive data)
  --capture-attention   record per-token region-pooled attention shares (requires --diagnostic-attention)
  --top-logprobs K      store the top-K token ids+logprobs and sampled-token surprisal per generated token
                        (default 0 = off; ~K*8 bytes/token)
  --prompt-scores       score PROMPT tokens during prefill (ent/m2/surprisal per position, stored under
                        prompt region ids)
  --spans-file SPANS_FILE
                        sections JSON [{"name","text"},...] attached to EVERY request, for section-level
                        attention under unmodified agent clients (per-request vitals.spans overrides it)
  --no-auto-sections    do not add automatic sections (system prompt by heading, tool:<name>, user:<k>,
                        self:<k>, tool_result:<tool>#<k>) to the parts of a request no declared span covers
  --reuse-kv            reuse KV across turns of an episode when the new prompt exactly extends the resident
                        tokens (append-only; kills quadratic re-prefill). NOTE: prefill activation capture
                        then covers only the delta rows
```

## vitals replay

```text
usage: vitals replay [-h] --store STORE --gguf GGUF [--n-ctx N_CTX] [--n-gpu-layers N_GPU_LAYERS]
                     [--lib LIB] [--episodes EPISODES] [--spans-file SPANS_FILE] [--no-auto-sections]
                     [--sidecar] [--force] [--channels CHANNELS] [--top-k TOP_K] [--no-token-attention]
                     [--section-diagnostics] [--chunk CHUNK] [--linear-reading]
                     [--linear-max-slots LINEAR_MAX_SLOTS] [--linear-group-sep LINEAR_GROUP_SEP]

options:
  -h, --help            show this help message and exit
  --store STORE         trace store directory (episodes)
  --gguf GGUF           the SERVING model's GGUF (attention must come from the same weights)
  --n-ctx N_CTX         context window per sequence in tokens (default 32768)
  --n-gpu-layers N_GPU_LAYERS
                        layers offloaded to the GPU (default: all)
  --lib LIB             libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
  --episodes EPISODES   comma-separated episode ids (default: all)
  --spans-file SPANS_FILE
                        sections JSON [{"name","text"},...] (e.g. from vitals annotate), resolved into
                        span:* regions during replay
  --no-auto-sections    do not add automatic sections to turns for which neither --spans-file nor the store
                        declares any
  --sidecar             write replay_* files instead of filling the live-capture channels (for replay-vs-
                        capture validation)
  --force               overwrite existing channels
  --channels CHANNELS   comma list: attention (fa-off, slower) and/or logits (ent/m2/surprisal/top-K, runs
                        fa-on at full speed)
  --top-k TOP_K         top-K next-token logprobs stored per forced token
  --no-token-attention  skip attn_tokens.parquet (the attention each prompt token receives; peaks, hottest
                        lines, the ledger and explanations need it)
  --section-diagnostics
                        capture exact per-step section peaks and attention-focus JS before sparse truncation
                        (section_diagnostics.parquet; needs the attention channel)
  --chunk CHUNK         forced tokens per engine call (default $VITALS_REPLAY_CHUNK, else 64; 256 is faster
                        when device memory allows)
  --linear-reading      hybrid models: also measure the linear-attention layers per section (retained,
                        memory share, read, retrieval) into linear_reading.parquet
  --linear-max-slots LINEAR_MAX_SLOTS
                        section memories per linear layer; beyond this, old message regions fold together
  --linear-group-sep LINEAR_GROUP_SEP
                        group declared spans by the part of their name before this separator (e.g. '/')
```

## vitals report

```text
usage: vitals report [-h] --store STORE [--json] [--out OUT] [--diagnostics]

options:
  -h, --help     show this help message and exit
  --store STORE  trace store directory (episodes)
  --json         print JSON rows instead of a table
  --out OUT      write the JSON rows here
  --diagnostics  include consult decomposition, both lift averages, and captured per-step peaks and focus JS
```

## vitals dashboard

```text
usage: vitals dashboard [-h] [--host HOST] [--port PORT] [--cache-dir CACHE_DIR] [--explain-dir EXPLAIN_DIR]
                        [--alarm-dir ALARM_DIR]
                        ROOT [ROOT ...]

positional arguments:
  ROOT                  trace stores, Meta-Harness run directories (logs/<run>), or directories to search
                        for them

options:
  -h, --help            show this help message and exit
  --host HOST           address to listen on (default 127.0.0.1)
  --port PORT           port (default 8040)
  --cache-dir CACHE_DIR
                        keep cached read maps here instead of beside each episode (default:
                        $VITALS_READMAP_CACHE, else the episode directory when writable)
  --explain-dir EXPLAIN_DIR
                        decision explanations written by vitals explain (default: $VITALS_EXPLAIN_DIR, else
                        explanations/ under the vitals cache, ~/.cache/vitals)
  --alarm-dir ALARM_DIR
                        flags and explanations written by vitals pipeline (default: $VITALS_ALARM_DIR, else
                        alarms/ under the vitals cache, ~/.cache/vitals)
```

## vitals annotate

```text
usage: vitals annotate [-h] [--by {heading,paragraph,marker}] [--template] [--out OUT]
                       [--clean-out CLEAN_OUT] [--check SECTIONS] [--gguf GGUF]
                       file

positional arguments:
  file                  the prompt file to annotate; with --check, the prompt file or episode directory to
                        check against

options:
  -h, --help            show this help message and exit
  --by {heading,paragraph,marker}
                        one section per heading block, per paragraph, or per <!-- section: name --> marker
  --template            the file is a prompt template: split sections at placeholders ({{ x }}, {% ... %},
                        {x}, ${x}) and declare only the fixed text
  --out OUT             write the sections JSON here (default: stdout)
  --clean-out CLEAN_OUT
                        with --by marker: write the prompt with the markers removed here
  --check SECTIONS      check this sections JSON against FILE instead of annotating
  --gguf GGUF           with an episode directory: render its last stored request with this model's chat
                        template (otherwise the message texts are checked)
```

## vitals monitor

```text
usage: vitals monitor [-h] --store STORE [--gguf GGUF] [--episodes EPISODES] [--budget BUDGET]
                      [--reference REFERENCE] [--cache-dir CACHE_DIR] [--out-dir OUT_DIR] [--json]

options:
  -h, --help            show this help message and exit
  --store STORE         trace store directory (episodes)
  --gguf GGUF           the model's GGUF, for its tokenizer only (no model is loaded; default: $VITALS_GGUF)
  --episodes EPISODES   comma-separated episode ids (default: all)
  --budget BUDGET       false-alarm budget: the share of argument values below the warn threshold
  --reference REFERENCE
                        reference store for section-drift alarms
  --cache-dir CACHE_DIR
                        read-map cache for drift (default: beside each episode)
  --out-dir OUT_DIR     alarm cache directory (default: $VITALS_ALARM_DIR, else alarms/ under the vitals
                        cache, ~/.cache/vitals)
  --json                print every alarm as JSON lines
```

## vitals ledger

```text
usage: vitals ledger [-h] --store STORE --gguf GGUF [--n-ctx N_CTX] [--n-gpu-layers N_GPU_LAYERS]
                     [--lib LIB] [--episodes EPISODES] [--turns TURNS] [--layers LAYERS] [--out-dir OUT_DIR]
                     [--force] [--no-writes] [--linear]

options:
  -h, --help            show this help message and exit
  --store STORE         trace store directory (episodes)
  --gguf GGUF           reference model GGUF (the model that produced the logged tokens)
  --n-ctx N_CTX         context window per sequence in tokens (default 65536)
  --n-gpu-layers N_GPU_LAYERS
                        layers offloaded to the GPU (default: all)
  --lib LIB             libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
  --episodes EPISODES   comma-separated episode ids (default: all)
  --turns TURNS         comma-separated turns, or all (every turn with an action onset)
  --layers LAYERS       comma-separated residual layers to capture (default: at 36%, 48%, 61% and 73% of the
                        model's depth; 23,31,39,47 on a 64-layer model)
  --out-dir OUT_DIR     ledger cache directory (default: $VITALS_LEDGER_DIR, else ledger/ under the vitals
                        cache, ~/.cache/vitals)
  --force               recapture decisions already in the cache
  --no-writes           skip the per-section writes (the second pass that lets a probe be split by section
                        from the ledger)
  --linear              also split the linear-attention layers' writes (Gated DeltaNet and similar)
```

## vitals probe train

```text
usage: vitals probe train [-h] --store STORE [--out-dir OUT_DIR] --labels LABELS --name NAME --layer LAYER
                          [--budget BUDGET] [--C C] [--normal-on {negatives,all}] --out OUT

options:
  -h, --help            show this help message and exit
  --store STORE         trace store the ledger was captured from
  --out-dir OUT_DIR     ledger cache directory (default: $VITALS_LEDGER_DIR, else ledger/ under the vitals
                        cache, ~/.cache/vitals)
  --labels LABELS       CSV with columns episode, turn, label (0/1) and optionally group (folds; default:
                        episode)
  --name NAME           behavior name
  --layer LAYER         residual layer, one of those the ledger captured (vitals ledger --layers)
  --budget BUDGET       false-alarm budget for the threshold
  --C C                 logistic regression regularization
  --normal-on {negatives,all}
                        decisions that form each section's normals (probe lift and read lift)
  --out OUT             probe JSON to write
```

## vitals probe score

```text
usage: vitals probe score [-h] --store STORE [--out-dir OUT_DIR] --probe PROBE [--episodes EPISODES]
                          [--top TOP]

options:
  -h, --help           show this help message and exit
  --store STORE        trace store the ledger was captured from
  --out-dir OUT_DIR    ledger cache directory (default: $VITALS_LEDGER_DIR, else ledger/ under the vitals
                       cache, ~/.cache/vitals)
  --probe PROBE        probe JSON (vitals probe train)
  --episodes EPISODES  comma-separated episode ids (default: all)
  --top TOP            sections listed per decision
```

## vitals pipeline

```text
usage: vitals pipeline [-h] --store STORE --gguf GGUF [--n-ctx N_CTX] [--n-gpu-layers N_GPU_LAYERS]
                       [--lib LIB] [--probes PROBES] [--layers LAYERS] [--reference REFERENCE] [--confirm]
                       [--webhook WEBHOOK] [--webhook-min {warn,alert}] [--watch WATCH]
                       [--episodes EPISODES] [--budget BUDGET] [--out-dir OUT_DIR] [--ledger-dir LEDGER_DIR]
                       [--no-replay]

options:
  -h, --help            show this help message and exit
  --store STORE         trace store directory (vitals import or vitals serve)
  --gguf GGUF           reference model GGUF (the model that produced the logged tokens)
  --n-ctx N_CTX         context window per sequence in tokens (default 32768)
  --n-gpu-layers N_GPU_LAYERS
                        layers offloaded to the GPU (default: all)
  --lib LIB             libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
  --probes PROBES       comma-separated probe JSON files (vitals probe train)
  --layers LAYERS       ledger residual layers (default: those of the probes, else the ledger default); a
                        probe's layer must be among them
  --reference REFERENCE
                        reference store for section-drift alarms
  --confirm             confirm probe flags by ablation and add probe peaks (GPU)
  --webhook WEBHOOK     URL to POST each new flag to (JSON)
  --webhook-min {warn,alert}
                        lowest severity sent
  --watch WATCH         seconds between passes (0: run once)
  --episodes EPISODES   comma-separated episode ids (default: all)
  --budget BUDGET       false-alarm budget of the argument checks
  --out-dir OUT_DIR     alarm and explanation cache (default: $VITALS_ALARM_DIR, else alarms/ under the
                        vitals cache, ~/.cache/vitals)
  --ledger-dir LEDGER_DIR
                        ledger cache (default: $VITALS_LEDGER_DIR, else ledger/ under the vitals cache,
                        ~/.cache/vitals)
  --no-replay           skip replay (stores captured live by vitals serve)
```

## vitals explain

```text
usage: vitals explain [-h] --store STORE --episode EPISODE [--turns TURNS] --gguf GGUF [--n-ctx N_CTX]
                      [--n-gpu-layers N_GPU_LAYERS] [--lib LIB] [--weights WEIGHTS] [--behavior BEHAVIOR]
                      [--ablate TOP_K] [--out-dir OUT_DIR] [--json | --agent]

options:
  -h, --help            show this help message and exit
  --store STORE         trace store directory (episodes)
  --episode EPISODE     episode id
  --turns TURNS         comma-separated turns, or all
  --gguf GGUF           reference model GGUF (the model that produced the logged tokens)
  --n-ctx N_CTX         context window per sequence in tokens (default 65536)
  --n-gpu-layers N_GPU_LAYERS
                        layers offloaded to the GPU (default: all)
  --lib LIB             libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
  --weights WEIGHTS     head weights JSON (vitals.analysis.calibration.save_weights); default: rank by the
                        read map
  --behavior BEHAVIOR   behavior map JSON (vitals.analysis.calibration.fit_behavior_map)
  --ablate TOP_K, --top-k TOP_K
                        sections ablated per decision
  --out-dir OUT_DIR     explanation cache directory (default: $VITALS_EXPLAIN_DIR, else explanations/ under
                        the vitals cache, ~/.cache/vitals)
  --json                print the full explanations as JSON
  --agent               print the compact form for agents (one JSON object per turn)
```

## vitals sensitivity

```text
usage: vitals sensitivity [-h] --store STORE --episode EPISODE --turn TURN --section SECTION --steps STEPS
                          --replacements REPLACEMENTS --gguf GGUF [--n-ctx N_CTX]
                          [--n-gpu-layers N_GPU_LAYERS] [--lib LIB] [--out OUT]

options:
  -h, --help            show this help message and exit
  --store STORE         trace store directory (episodes)
  --episode EPISODE     episode id
  --turn TURN           turn index
  --section SECTION     declared section name (with or without span:)
  --steps STEPS         comma-separated output positions, e.g. 0,5,12
  --replacements REPLACEMENTS
                        JSON file mapping condition names to text or {token_ids: [...]}
  --gguf GGUF           reference model GGUF (the model that produced the logged tokens)
  --n-ctx N_CTX         context window per sequence in tokens (default 32768)
  --n-gpu-layers N_GPU_LAYERS
                        layers offloaded to the GPU (default: all)
  --lib LIB             libllama path (default: $VITALS_LLAMA_LIB, else the pinned build)
  --out OUT             write the JSON result here
```

## vitals sections

```text
usage: vitals sections [-h] [--min-errors MIN_ERRORS] [--min-correct MIN_CORRECT] [--top TOP]
                       [--match {exact,contains}]
                       root

positional arguments:
  root                  a store dir or a replay root of <name>/store dirs

options:
  -h, --help            show this help message and exit
  --min-errors MIN_ERRORS
                        minimum rows (episode turns) from failed episodes per section contrast
  --min-correct MIN_CORRECT
                        minimum rows (episode turns) from successful episodes per section contrast
  --top TOP             sections listed per dataset
  --match {exact,contains}
                        label-to-answer matching for the gradient
```

## vitals gate

```text
usage: vitals gate [-h] --gguf GGUF --prompt-file PROMPT_FILE [--n N] [--n-ctx N_CTX] [--out OUT] {fa}

positional arguments:
  {fa}                  the gate to run

options:
  -h, --help            show this help message and exit
  --gguf GGUF           model GGUF
  --prompt-file PROMPT_FILE
                        text file with the prompt to continue
  --n N                 greedy tokens to compare
  --n-ctx N_CTX         context window in tokens
  --out OUT             write the JSON report here
```

## vitals calibrate

```text
usage: vitals calibrate [-h] --gguf GGUF [--n-ctx N_CTX] [--out OUT]
                        {retrieval-heads,induction-heads,position-bias}

positional arguments:
  {retrieval-heads,induction-heads,position-bias}
                        the census to run

options:
  -h, --help            show this help message and exit
  --gguf GGUF           model GGUF
  --n-ctx N_CTX         context window in tokens
  --out OUT             write the JSON report here
```

## vitals engine doctor

```text
usage: vitals engine doctor [-h] [--lib LIB]

options:
  -h, --help  show this help message and exit
  --lib LIB   path to libllama (default: $VITALS_LLAMA_LIB or vendor/)
```

## vitals metaharness

The Meta-Harness add-on has its own commands: `vitals metaharness --help` and the [add-on guide](../../vitals/addons/metaharness/README.md).
