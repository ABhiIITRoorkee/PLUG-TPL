<h1>PLUG-TPL: A Modular LLM-Augmented Framework for Function-Aware Library Recommendation</h1>

<p><strong>Function-aware library recommendation for Android apps using dependency context and description-derived signals.</strong></p>

<h2>Overview</h2>
<p>
PLUG-TPL is a modular framework for Android third-party library (TPL) recommendation that combines
observed dependency context with description-derived evidence through a shared ranking pipeline.
The framework is organized into three <strong>Phases</strong> that use the same downstream ranker while differing
in how the app description is represented.
</p>

<h2>Phases in PLUG-TPL</h2>
<ul>
  <li><strong>Phase A (Context Signal)</strong> — Uses only observed library co-usage/context information.</li>
  <li><strong>Phase B (Semantic Signal)</strong> — Adds a raw-description semantic intent vector.</li>
  <li><strong>Phase C (Verified Function Signal)</strong> — Mines function statements from descriptions, verifies textual support, and pools verified functions into a function-aware representation.</li>
</ul>

<h2>Features</h2>
<ul>
  <li><strong>Modular Three-Phase Design</strong> with a shared scoring and ranking pipeline.</li>
  <li><strong>Context-Aware Recommendation</strong> from observed app libraries.</li>
  <li><strong>Description-Aware Signal Fusion</strong> for intent-aware ranking.</li>
  <li><strong>LLM-Augmented Function Mining</strong> with verification for more reliable text-derived evidence.</li>
  <li><strong>Single-Command Pipeline Execution</strong> via <code>main.py</code> (train → test → metrics).</li>
</ul>

<h2>Requirements</h2>
<ul>
  <li>Python 3.9+ (recommended)</li>
  <li>PyTorch (GPU optional, CUDA recommended for faster runs)</li>
  <li>Dependencies from <code>requirements.txt</code></li>
</ul>

<h2>Setup</h2>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>Run</h2>
<p>Run the full pipeline (train → test → metrics) for all folds and all <code>rm</code> values:</p>

<pre><code>python main.py --mode all --folds 0,1,2,3,4 --rms 1,3,5 --device cuda</code></pre>

<h2>Outputs</h2>
<ul>
  <li>Model checkpoints are saved during training.</li>
  <li>Recommendation files are generated during testing.</li>
  <li>Final metric summaries are produced during the metrics step (as configured in your pipeline).</li>
</ul>
