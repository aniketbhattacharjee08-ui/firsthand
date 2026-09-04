# 08 — The Reliability Engineering of a Near-100% "Human" Pass Rate

*Deep research report, 2026-09-03. Companion to reports 01–06. This report deliberately contains almost no prompt engineering. It is about the systems and statistics layer that sits on top of whatever rewriter you build: how many samples, judged by what, with what guarantee, at what cost, and how you know it still works next month.*

---

## 0. The thesis in one paragraph

A humanizer with a 60% per-attempt pass rate and a good reliability layer beats a humanizer with an 85% per-attempt pass rate and no reliability layer. But the reliability layer has three hard ceilings that most builders discover only in production: (1) **candidate correlation** — the N samples fail together, so the exponential `1-(1-p)^N` you budgeted for is really a power law; (2) **verifier quality** — you can never exceed the agreement rate between your local judge and the real detector, and best-of-N actively *seeks out* the places where they disagree; (3) **non-exchangeability** — every statistical guarantee you compute dies the day GPTZero ships a new model. Everything below is about quantifying and engineering around those three.

---

## 1. Best-of-N / rejection sampling: the math, and where it stops being true

### 1.1 The textbook formula and its correct estimator

If each attempt independently passes with probability `p`, and you keep sampling until one passes (or up to N):

```
P(at least one pass in N)  =  1 - (1-p)^N
E[attempts until first pass] = 1/p        (geometric, unbounded N)
E[attempts | capped at N]    = (1 - (1-p)^N) / p
```

When you *measure* this from n sampled candidates of which c passed, do not plug `p̂ = c/n` into `1-(1-p̂)^k` — that is biased. Chen et al.'s Codex paper gives the unbiased estimator, which is the standard in this literature ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)):

```
pass@k = E_docs [ 1 - C(n-c, k) / C(n, k) ]
```

### 1.2 The two things that break independence

**(a) Heterogeneity across documents.** Real pass probability is a function of the input, `p(x)`. The fleet-level pass rate is `E_x[1 - (1-p(x))^N]`, and it saturates at `1 - P(p(x) = 0)`. If 1.5% of your inputs are documents your rewriter simply cannot make pass (dense technical prose, heavy citation, very short text — report 03 found academic/technical humanizes worst at 4/10 quality), then **no N gets you past 98.5%**. This is the single most important number in the whole system and it is not visible in an average.

**(b) Correlation across candidates for the same document.** N samples from one model on one prompt are not independent draws — they share a prompt, a checkpoint, and a mode. Model the per-document pass probability as `p(x) ~ Beta(a,b)` with mean `μ = a/(a+b)` and intra-class correlation `ρ = 1/(a+b+1)`. Then:

```
P(all N candidates fail) = Π_{i=0}^{N-1} (b+i)/(a+b+i)
```

Worked numbers at `μ = 0.6`:

| ρ (candidate correlation) | P(pass) at N=8 | Failure rate vs. independent |
|---|---|---|
| 0 (independent) | 99.93% | 1× |
| 0.2 | 97.50% | 38× worse |
| 0.5 | 87.25% | 195× worse |

And the asymptotics matter more than the point values: for large N, `P(all fail) ≈ C·N^(-a)`. **Correlated best-of-N decays as a power law in N, not exponentially.** This is exactly the empirical signature Hughes et al. report for Best-of-N Jailbreaking, where attack success rate follows "power-law-like behavior for many orders of magnitude" of N ([arXiv:2412.03556](https://arxiv.org/abs/2412.03556)) — 89% ASR on GPT-4o but only at N=10,000. Treat any observed power law in your own N-sweep as a direct measurement of `a`, and therefore of ρ.

The mechanism is documented directly: "wider sampling alone can suffer from diminishing returns: new rollouts often repeat existing answer patterns instead of adding useful reasoning diversity" ([arXiv:2608.05643](https://arxiv.org/abs/2608.05643)). And the modern instruction-tuned checkpoint you are sampling from is the *worst possible* base for decorrelated sampling: RLHF "substantially decreases the variety of model outputs" relative to SFT ([arXiv:2310.06452](https://arxiv.org/abs/2310.06452)); RLVR models beat base models at k=1 but are *overtaken at large k* because the sampling boundary narrows ([arXiv:2504.13837](https://arxiv.org/abs/2504.13837)); and mode collapse traces to typicality bias in preference data ([arXiv:2510.01171](https://arxiv.org/abs/2510.01171)). This is the statistical restatement of synthesis finding #4: the instruct fingerprint and the sampling-diversity collapse have the same cause.

### 1.3 Buying decorrelation

Ranked by evidence strength for *this* problem:

1. **Raise temperature as N grows.** Codex measured optimal `T* = 0.2` for pass@1 and `T* = 0.8` for pass@100, explicitly "due to the increased sample diversity" ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)). Sweep temperature *jointly with N*, never separately.
2. **Diversify the input, not just the sampler.** The transfer-attack literature is unambiguous that random input transformations at each iteration materially improve black-box transfer (DI-2-FGSM, +6.6% over the NIPS-2017 winner, [arXiv:1803.06978](https://arxiv.org/abs/1803.06978)); BoN Jailbreaking's entire mechanism is augmentation diversity, not sampler diversity. For a humanizer this means: vary the *plan* (target sentence-length profile, register, opening move, paragraph function order) per candidate, drawn from the human reference distribution of report 04. Different control codes → genuinely different failure modes.
3. **Diverse decoding.** Diverse Beam Search optimizes an explicitly diversity-augmented objective with negligible overhead ([arXiv:1610.02424](https://arxiv.org/abs/1610.02424)); nucleus sampling remains the quality/diversity workhorse ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)). DVTS (Diverse Verifier Tree Search) splits the beam budget into independent subtrees precisely to stop a verifier-guided search collapsing onto one mode.
4. **Multi-model / multi-checkpoint pools.** Two 4B rewriters from different base families give lower ρ than 16 samples from one. This is the cheapest large win available and almost nobody does it.
5. **Calibrated sampling.** CarBoN fits an input-specific temperature and logit shift and reports "up to 4× reduction in required rollouts" for equal accuracy ([arXiv:2510.15674](https://arxiv.org/abs/2510.15674)); Soft Best-of-n interpolates to the tilted distribution with an O(1/n) convergence guarantee ([arXiv:2505.03156](https://arxiv.org/abs/2505.03156)). These are variance-reduction techniques: same pass rate, fewer draws.
6. **Fuse instead of select.** Fusion-of-N argues selection is "inherently zero-sum" and synthesizes across candidates instead ([arXiv:2510.00931](https://arxiv.org/abs/2510.00931)). For us this is natural: take the best paragraph from candidate 3 and the best from candidate 7. It also breaks correlation, because the fused document is not in the sampling support of any single draw.

### 1.4 Cost per document

Let `c_g` = cost of one rewrite, `r_τ` = fraction of candidates the local surrogate accepts at threshold τ, `q_τ` = P(real detector passes | surrogate accepted), `c_v` = cost of one live detector call.

```
Surrogate-only:      cost = c_g / r_τ                     pass rate = q_τ
Surrogate + verify:  cost = (1/q_τ) · (c_g/r_τ + c_v)     pass rate = 1 - (1-q_τ)^M
```

Concrete: `c_g = $0.010` (1,000-word rewrite, mid-tier model), `r_τ = 0.5`, `q_τ = 0.8`, `c_v = $0.065` (Originality.ai: 1 credit = 100 words, Pro plan $12.95 for 2,000 credits → $0.0065/credit → 10 credits for 1,000 words; [originality.ai/pricing](https://originality.ai/pricing)). Then cost ≈ **$0.106/document** and residual failure after M=4 verify rounds is 0.16% *if independent* — and ~2–4% at ρ=0.2–0.3. **Live verification, not more sampling, is what buys the last few points**, and it costs roughly 6× the generation itself. Budget accordingly.

### 1.5 When to stop adding N

Stroebl, Kapoor and Narayanan give the crisp answer: with an imperfect verifier, "resampling cannot decrease [the false-positive] probability, so it imposes an upper bound to the accuracy of resampling-based inference scaling, regardless of compute budget," and empirically "optimal sampling attempts are often fewer than 10" ([arXiv:2411.17501](https://arxiv.org/abs/2411.17501)). Brown et al. observe the same from the selection side: coverage keeps scaling but "majority voting and reward models plateau beyond several hundred samples" ([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)).

---

## 2. The verifier is the ceiling

### 2.1 Best-of-N is an optimizer, and every optimizer Goodharts

Gao, Schulman and Hilton fit closed-form overoptimization laws with `d = √(D_KL(π‖π_init))` ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)):

```
Best-of-N:  R_bon(d) = d(α_bon - β_bon·d)
RL:         R_RL(d)  = d(α_RL  - β_RL·log d)
KL_bon      = log n - (n-1)/n
```

The quadratic in `d` means gold-reward performance **turns over and declines** as you push N. `β` shrinks with reward-model size, so a bigger/better verifier buys you a longer runway — nothing more. Beirami et al. later showed `log n - (n-1)/n` is an *upper bound*, not the exact KL, and that best-of-n win rate against the reference is bounded by `n/(n+1)` ([arXiv:2401.01879](https://arxiv.org/abs/2401.01879)). Skalse et al. prove the general impossibility: for the set of all stochastic policies, two reward functions are unhackable only if one is constant ([arXiv:2209.13085](https://arxiv.org/abs/2209.13085)). Rafailov et al. show even preference-optimization methods with no explicit proxy degrade "across a wide range of KL budgets" ([arXiv:2406.02900](https://arxiv.org/abs/2406.02900)). Translation for us: **your local detector will be beaten, and the text that beats it hardest is the text most likely to look weird to GPTZero.**

### 2.2 "My local detector says human with p=0.99 — what is P(GPTZero says human)?"

There is no derivation. The 0.99 is a within-model posterior over the local model's own training distribution; it carries no information about a different classifier trained on different data. The only correct object is the *joint* distribution of (local score, GPTZero verdict) **measured on your own output distribution**:

```
q_τ = P(GPTZero = human | local score ≥ τ, x ~ my humanizer's outputs)
```

Two corrections stack on top:

- **Calibration.** Modern nets are badly overconfident; temperature scaling (single-parameter Platt) is the recommended fix ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Fit it on held-out data and report ECE. But calibrating your local detector only fixes `P(local's own label)`, not `q_τ`.
- **Selection bias.** Best-of-N does not sample a random candidate at score τ — it takes the argmax. So the operative quantity is `q` conditioned on being at the extreme of the surrogate, where surrogate/target disagreement is by construction largest. Empirically, `q_argmax < q_τ`, and the gap widens with N. This is the formal version of Stroebl's ceiling.

**Practical rule:** publish a single number, `q_τ`, measured against the live detector on a stratified canary set, with a Clopper–Pearson interval. Never surface the local detector's probability to a user.

### 2.3 Getting an actual guarantee: conformal methods

Split conformal prediction converts "I have a score" into "I have a distribution-free finite-sample coverage guarantee," and there is now a direct line of work on classifier-in-the-loop generation:

- **Conformal Language Modeling** ([arXiv:2306.10193](https://arxiv.org/abs/2306.10193)) is *literally our architecture with a proof attached*: it calibrates a **stopping rule** (when to stop drawing candidates) and a **rejection rule** (which candidates to discard) so that "the sampled set returned by our procedure contains at least one acceptable answer with high probability." Substitute "acceptable" = "GPTZero says human" and you have a principled way to choose both N and τ.
- **Learn Then Test** ([arXiv:2110.01052](https://arxiv.org/abs/2110.01052)) reframes hyperparameter choice as multiple hypothesis testing, giving finite-sample risk control for *any* black-box pipeline — the right tool for picking τ, N, and the number of refinement rounds simultaneously.
- **Conformal Risk Control** ([arXiv:2208.02814](https://arxiv.org/abs/2208.02814)) controls `E[loss] ≤ α` for any monotone loss, e.g. "expected fraction of AI-flagged sentences ≤ 5%."
- **Conformal factuality** ([arXiv:2402.10978](https://arxiv.org/abs/2402.10978)) gives the back-off pattern: when confidence is short, make the claim less specific. Our analogue is graceful degradation — return the document plus an honest "this one is borderline" flag rather than shipping a false promise.

**The catch, stated plainly:** all of these guarantee *marginal* coverage under *exchangeability* between calibration and test data. A GPTZero model update, a new input genre, or a change to your own rewriter all break exchangeability and void the guarantee. Weighted conformal prediction handles known covariate shift via likelihood-ratio weights ([arXiv:1904.06019](https://arxiv.org/abs/1904.06019)), but you rarely know the shift. So: **conformal gives you a defensible number between recalibrations; the monitoring system in §6 is what tells you when to recalibrate.**

### 2.4 Cascades and deferral

The right economic structure is a cascade: free local detectors → cheap open ensemble → paid live API, escalating only on uncertainty. FrugalGPT reports up to 98% cost reduction at matched quality with a learned scoring/deferral function ([arXiv:2305.05176](https://arxiv.org/abs/2305.05176)); mixture-of-thought cascades hit GPT-4 parity at 40% of cost using answer-consistency as the deferral signal ([arXiv:2310.03094](https://arxiv.org/abs/2310.03094)). But Jitkrittum et al. warn that naive confidence-based deferral fails in exactly our three conditions — **specialist downstream models, label noise, and distribution shift** — and that post-hoc learned deferral rules do substantially better ([arXiv:2307.02764](https://arxiv.org/abs/2307.02764)). Our downstream model (GPTZero) is precisely a specialist under shift. So learn the deferral rule on measured (local score, GPTZero verdict) pairs; do not threshold the local probability.

---

## 3. Ensembles and transfer to a held-out commercial detector

### 3.1 The prior from adversarial ML

Attacking an *ensemble* of white-box models is the canonical way to produce examples that transfer to an unseen black box — Liu et al. were the first to get targeted transfer at all, and they broke commercial Clarifai with it ([arXiv:1611.02770](https://arxiv.org/abs/1611.02770)). Combining ensembles with input diversity compounds the effect ([arXiv:1803.06978](https://arxiv.org/abs/1803.06978)). This is the strongest theoretical reason to believe detector-ensemble optimization transfers.

### 3.2 The measured transfer gaps in AI-text detection

| Work | Trained/guided against | Result | Transfer note |
|---|---|---|---|
| Adversarial Paraphrasing ([arXiv:2506.07001](https://arxiv.org/abs/2506.07001)) | OpenAI-RoBERTa-Large as guide, training-free | **−87.88% mean T@1%F** across detectors | Enormous spread: **−98.96% on Fast-DetectGPT vs −64.49% on RADAR** |
| AuthorMist ([arXiv:2503.08716](https://arxiv.org/abs/2503.08716)) | GPTZero, WinstonAI, Originality as RL reward | **78.6%–96.2% ASR** per detector, semantic sim >0.94 | 18-point spread across detectors |
| StealthRL ([arXiv:2602.08934](https://arxiv.org/abs/2602.08934)) | RoBERTa, Fast-DetectGPT, Binoculars, MAGE (GRPO on Qwen3-4B) | 97.6% ASR | "attacks transfer to two held-out detectors… revealing shared architectural vulnerabilities" |
| SICO ([arXiv:2305.10847](https://arxiv.org/abs/2305.10847)) | in-context substitution | AUC −0.5 average across **six** detectors | Broad but shallow |

The headline is that transfer is **real but wildly uneven**: a 34-point spread within a single paper. A mean improvement number is therefore actively misleading for a pass/fail product.

### 3.3 How many detectors, and which

Correlation, not count, is what matters. Goel et al. show that as models get more capable their *mistakes converge* — measured by chance-adjusted probabilistic agreement (CAPA) — which "undermines AI oversight" because ensemble members fail on the same items ([arXiv:2502.04313](https://arxiv.org/abs/2502.04313)). Three RoBERTa-derived detectors are close to one detector. The correct portfolio is **one member per detection family**, matching report 01's taxonomy:

1. a token-surprisal/zero-shot detector (Fast-DetectGPT or Binoculars),
2. a fine-tuned transformer classifier (the e5-small-lora or a RoBERTa detector),
3. an authorship/style-embedding detector (StyleDistance/LUAR) — the one Rivera Soto et al. showed still catches detector-RL'd text at 0.95 AUROC,
4. a paraphrase-aware / humanizer-trained detector proxy, since GPTZero, Turnitin and Pangram all now ship one,
5. optionally a live commercial call, used sparsely as ground truth rather than as a training signal.

Complementary ensembling does measurably help detection robustness under shift — DoGEN's domain-gated weighting "outperform[s] models twice its size on out-of-domain detection" ([arXiv:2505.13855](https://arxiv.org/abs/2505.13855)); confidence-weighted ensembles are the recommended mitigation for the distribution-shift failure of a strong single baseline ([arXiv:2607.03680](https://arxiv.org/abs/2607.03680)); MOSAIC ensembles multiple observer LLMs for robustness ([arXiv:2409.07615](https://arxiv.org/abs/2409.07615)). Everything that makes a *detector* ensemble more robust makes it a *better proxy* for a held-out commercial detector — which is what we want in the reward.

### 3.4 Averaging vs. max-over-detectors

The deployment metric is a gate: the document ships only if the real detector passes it. The analogous training objective is therefore **worst-case, not average** — minimize `max_d score_d(y)`, because averaging lets a huge win on Fast-DetectGPT paper over a fatal loss on the classifier. This is the standard min-max formulation of ensemble attack (black-box min-max frameworks, [arXiv:1909.13806](https://arxiv.org/abs/1909.13806)); its benefit is exactly that it doesn't let one easy member absorb the objective.

The practical caveat: a hard max has sparse, high-variance gradients and lets the single hardest detector dominate. The recommended compromise, and the one worth A/B-testing first:

```
L = (1/λ) · log Σ_d exp(λ · score_d(y))        # log-sum-exp: λ→0 gives mean, λ→∞ gives max
```
Train with moderate λ (soft-max over detectors), **but always gate on the hard max**, and always report per-detector numbers rather than the mean. Also sample which detector is active per rollout (report 05's Stage-2 design) — that is stochastic max, and it decorrelates the reward the same way input diversity decorrelates candidates.

---

## 4. Sentence-level surgical editing: attractive, and mostly a trap

### 4.1 What GPTZero actually does

GPTZero runs both a sentence classifier and a document-level model, returns `document_classification` ∈ {HUMAN_ONLY, MIXED, AI_ONLY} plus `class_probabilities`, `confidence_category` and `highlight_sentence_for_ai` ([gptzero.me/technology](https://gptzero.me/technology)). The Advanced Scan labels sentences high/medium/low *impact*: "Sentences with high impact are very likely to be adding to our confidence in your AI/human probability score." And then, decisively: **"You may find that adjusting an entire document changes your score more than just adjusting the sentences. This is because our detector holistically analyzes the document."** ([GPTZero support](https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning)).

So the aggregation is **not** a mean and **not** a max — it is a learned document-level function over sentence representations plus global features. Which means targeted repair is coordinate descent on a **non-separable** objective.

### 4.2 The arithmetic, under each aggregation hypothesis

Let the document have `n` sentences; you repair `k` of them from `s_hi` to `s_lo`.

- **Mean aggregation:** `Δ = (k/n)(s_hi - s_lo)`. With n=40, k=5, 0.95→0.15: **Δ = 0.10**. To drag a 0.85 document to 0.15 you must fix ~35 of 40 sentences. This single calculation explains GPTZero's own advice quantitatively.
- **Max aggregation:** fixing the single worst sentence moves the score to the second-worst. Repair is then a whack-a-mole with n rounds in the worst case.
- **Learned aggregation (reality):** unknown and non-monotone. Empirically sublinear in k, with the added hazard below.

### 4.3 The two caveats that kill naive greedy repair

**Context dependence.** Sentence scores are produced with surrounding context. Rewriting sentence *i* changes the scores of *i-1* and *i+1*, so the saliency ranking you computed is stale after one edit. The correct loop is: re-score the whole document after **every** edit, and treat the ranking as advisory only. This is exactly TextFooler's structure — deletion-based word-importance ranking, then substitution, then re-query the victim model ([arXiv:1907.11932](https://arxiv.org/abs/1907.11932)) — and the CondBERT detoxification pattern of classifier-locating the offending span and replacing only that span ([arXiv:2109.08914](https://arxiv.org/abs/2109.08914)). Both re-score after each edit; neither trusts a precomputed ranking.

**Variance destruction.** Report 01 records that burstiness — the *standard deviation* of per-sentence perplexity — is one of GPTZero's seven indicators, and report 04 puts spread (not level) as the single strongest human/AI separator. Greedily repairing only the highest-scoring sentences **reduces the variance of the sentence-score profile**, pushing the document toward the uniform-medium profile that is itself the AI signature. Targeted repair can lower every sentence score and *raise* the document score. This is a real, mechanistic reason to distrust the greedy loop.

### 4.4 The usable version

Use sentence scores for **localization and budgeting**, not as the edit unit:

1. Score the document; take the top-`k` high-impact sentences.
2. Rewrite the **paragraphs containing them**, not the sentences — restores context coherence and preserves the ability to re-introduce length variance.
3. Re-score the whole document; keep the edit only if the document score improved *and* all quality gates hold.
4. Cap at 2–3 rounds (§5), then fall back to a full-document regeneration with a different plan.

---

## 5. Iterative refinement with a critic: how many rounds, and how it rots

### 5.1 The convergence curve is known and it is steep-then-flat

Self-Refine caps at **4 iterations** and reports per-iteration scores that visibly saturate: Code Optimization 22.0 → 27.0 → 27.9 → 28.8; Sentiment Reversal 33.9 → 34.9 → 36.1 → 36.8; Constrained Generation 29.0 → 40.3 → 46.7 → 49.7 — with explicit "diminishing returns in the improvement as the number of iterations increases" ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)). Roughly **60–80% of the total gain lands in round 1**. Budget 2 rounds, allow a 3rd only when the gate is close.

### 5.2 The loop only works because the critic is external

This is the most robust finding in the self-correction literature and it directly justifies putting a detector in the loop:

- Huang et al.: "LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction" ([arXiv:2310.01798](https://arxiv.org/abs/2310.01798)).
- Stechly/Valmeekam/Kambhampati: self-critique *reduces* performance; independent sound verifiers substantially improve it, and simply re-prompting with verifier feedback captures most of the gain ([arXiv:2402.08115](https://arxiv.org/abs/2402.08115)).
- CRITIC's whole thesis is verify-then-correct with *tools*, and it concludes external feedback is "crucial" ([arXiv:2305.11738](https://arxiv.org/abs/2305.11738)).
- Reflexion works because it stores verbal feedback from an external environment signal in episodic memory ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)).
- Zhao et al. find frontier models have "remarkably weak out-of-the-box verification capabilities" ([arXiv:2502.01839](https://arxiv.org/abs/2502.01839)); training a generative verifier is what closes the generator–verifier gap (GSM8K best-of-N 73% → 93.4%, [arXiv:2408.15240](https://arxiv.org/abs/2408.15240)).

**So: never let the rewriter grade itself. The detector ensemble + the stylometric feature distance is the critic.** And because that critic is imperfect (§2), the loop is a bounded-round optimizer against a proxy, not a search for truth.

### 5.3 Compounding drift, and the four mechanisms that stop it

Each round is a paraphrase of a paraphrase. That is the inference-time analogue of model collapse, where recursive self-consumption makes "tails of the original content distribution disappear" ([arXiv:2305.17493](https://arxiv.org/abs/2305.17493)) — and it matches report 03's empirical finding that the best evaders win by factual drift, flattened register and broken collocations. Four mitigations, in priority order:

1. **Keep-best archive.** Maintain the best-scoring-and-gate-passing candidate seen so far; the loop can never ship something worse than round 0. This makes the whole procedure monotone by construction and is the single highest-value line of code in the system.
2. **Anchor to the *original*, never to the previous round.** Compute NLI entailment, embedding similarity, entity/number overlap and length ratio against `y_0` at every round. Chaining checks round-to-round permits unbounded drift by many small legal steps.
3. **Hard gates, not soft weights** (report 05, Stage 2). A round that violates a gate is *reverted*, not penalized. Soft weights are what produced StealthRL's 3.78 → 2.51 quality collapse.
4. **Breadth before depth.** "Refining over resampling" is best read as *both*: sample a few decorrelated candidates, then refine each shallowly, then select ([arXiv:2608.05643](https://arxiv.org/abs/2608.05643)). A 4×2 grid (4 seeds, 2 refine rounds) beats 1×8 and beats 8×1 for the same budget, because it attacks correlation and depth at once.

---

## 6. Guarantees and monitoring in production

### 6.1 The number you may honestly show a user

Not the local detector's probability. The honest number is the **measured pass rate on a stratified canary set against the live detector, with a confidence interval.**

- Use Wilson or Clopper–Pearson intervals for a binomial proportion, not the normal approximation ([NIST handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)).
- **Rule of three:** with zero failures in n trials, the 95% upper bound on the failure rate is `3/n`. To claim ≤1% failure you need **300 consecutive clean canary documents**; for ≤0.1%, 3,000 ([rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(statistics))). This is the correct way to price a "99%+" claim.
- Stratify the canary by genre, length, source model and (critically) the hard tail of §1.2. A single blended number hides the 1.5% of inputs that never pass.
- Ship a **conditional** promise: "97.8% of documents like yours passed GPTZero last week (95% CI 96.4–98.8%); if yours doesn't, we re-run free." That is defensible; "100% undetectable" is not, and report 03 shows every vendor making that claim is measurably wrong.

### 6.2 A/B testing against a live detector

To distinguish a 97% pipeline from a 99% pipeline at α=0.05, power 0.8, the two-proportion sample size is

```
n = [ z_{α/2}√(2p̄q̄) + z_β√(p₁q₁ + p₂q₂) ]² / (p₁-p₂)²   ≈ 770 documents per arm
```

At ~$0.065/document of live verification that is ~$100 per comparison — cheap, but it means you cannot A/B ten variants a day. Two consequences: (a) run **offline replays** on a frozen canary set for most decisions and reserve live A/B for finalists; (b) because you *will* peek at the results early, use always-valid p-values / sequential tests rather than fixed-horizon ones — continuous monitoring of a fixed-horizon test makes inferences "wholly unreliable" ([arXiv:1512.04922](https://arxiv.org/abs/1512.04922)).

### 6.3 Drift monitoring when the detector updates

Report 06 documents at least four significant GPTZero model updates in 2025–2026. Every one of them invalidates: your conformal calibration, your τ, your published pass rate, and possibly your entire technique. Detect it, don't discover it from support tickets:

- **Frozen reference battery.** ~200 documents that never change: known-human (pre-2022, multiple genres), known-AI (raw model output, several generators), and known-humanized (your own historical outputs). Score all of them against the live API daily. Track the *score distribution*, not the verdict — you want to see the mean shift before it crosses the threshold.
- **Alert on distributional change**, using KS / PSI on the score vector, which is standard production drift practice ([arXiv:2007.06299](https://arxiv.org/abs/2007.06299)); adversarial-validation drift detection (train a classifier to distinguish last month's scores from this month's; AUC ≫ 0.5 = drift) is the sharper version ([arXiv:2004.03045](https://arxiv.org/abs/2004.03045)).
- **Auto-degrade the published number** to its lower confidence bound the moment the battery shifts, and trigger recalibration of τ and N.
- Remember the base-rate direction: independent testing of 14 detection tools found them "neither accurate nor reliable," with a bias toward calling text human ([arXiv:2306.15666](https://arxiv.org/abs/2306.15666)). Vendors correct that bias over time — meaning drift is systematically *against* you.

### 6.4 Canarying your own releases

Google SRE's canarying methodology maps cleanly: a canary is "a partial and time-limited deployment of a change in a service and its evaluation"; size it so a bad release consumes little error budget (a 5% canary at 20% errors costs 1% overall); pick fewer than a dozen attributable SLI-based metrics; never compare before/after periods (time-of-day confounds); auto-roll-back on divergence ([sre.google/workbook/canarying-releases](https://sre.google/workbook/canarying-releases/)). Our SLIs: live pass rate, gate-violation rate, mean refinement rounds, cost per shipped document, and LLM-judge quality delta vs. the input.

---

## 7. Technique → expected lift → cost → evidence

Lift is expressed as change in *document pass rate*, from a nominal 60% single-shot baseline. "Cost" is per shipped document.

| Technique | Expected lift in pass rate | Cost | Evidence |
|---|---|---|---|
| Best-of-N, N=8, **independent** samples | 60% → 99.9% (theoretical) | 8× generation | `1-(1-p)^N` |
| Best-of-N, N=8, ρ≈0.2 (realistic single-model) | 60% → **97.5%** | 8× generation | Beta-binomial; power-law ASR in BoN jailbreaking ([2412.03556](https://arxiv.org/abs/2412.03556)) |
| Best-of-N, N=8, ρ≈0.5 (low-temp instruct model) | 60% → **87%** | 8× generation | RLHF diversity collapse ([2310.06452](https://arxiv.org/abs/2310.06452)), RLVR pass@k ([2504.13837](https://arxiv.org/abs/2504.13837)) |
| Raise temperature with N (T 0.2→0.8) | Recovers much of the ρ penalty; several points | free | Codex T* by k ([2107.03374](https://arxiv.org/abs/2107.03374)) |
| Per-candidate plan/control-code diversity | Large ρ reduction; the main lever | free | Input diversity transfer ([1803.06978](https://arxiv.org/abs/1803.06978)), BoN jailbreak augmentations |
| Two base models instead of one | Meaningful ρ reduction | infra only | Correlated-error argument ([2502.04313](https://arxiv.org/abs/2502.04313)) |
| Calibrated sampling (CarBoN / soft BoN) | Same pass rate at ~4× fewer rollouts | small | [2510.15674](https://arxiv.org/abs/2510.15674), [2505.03156](https://arxiv.org/abs/2505.03156) |
| Fusion-of-N instead of selection | Modest lift + breaks correlation | +1 LLM call | [2510.00931](https://arxiv.org/abs/2510.00931) |
| Local surrogate gate only (no live check) | Caps at `q_τ` (est. 70–85%) | ~$0.02 | Verifier ceiling ([2411.17501](https://arxiv.org/abs/2411.17501)) |
| **Live detector verify-then-retry, M=4** | `q_τ`=0.8 → **98–99.8%** (ρ-dependent) | ~$0.11 | Cascade economics ([2305.05176](https://arxiv.org/abs/2305.05176)) |
| Detector-**ensemble** objective (4 families, max-gate) | Raises `q_τ` by ~10–20 pts vs. single-detector | 4× local scoring | Ensemble transfer ([1611.02770](https://arxiv.org/abs/1611.02770)); AI-text spread of 34 pts ([2506.07001](https://arxiv.org/abs/2506.07001)) |
| Adding a 3rd RoBERTa-family detector | ≈0 | 1× scoring | Correlated errors ([2502.04313](https://arxiv.org/abs/2502.04313)) |
| Sentence-only surgical repair (top-5 of 40) | Δ≈0.10 doc score at best; can be **negative** | 5 rewrites | GPTZero: whole-document > sentence ([support](https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning)) |
| Paragraph-level targeted repair w/ full re-score | Moderate; useful as round 2 | 2–3 rewrites + re-scores | TextFooler loop ([1907.11932](https://arxiv.org/abs/1907.11932)), CondBERT ([2109.08914](https://arxiv.org/abs/2109.08914)) |
| Refinement round 1 | ~60–80% of all refinement gain | 1 rewrite | Self-Refine Fig. 4 ([2303.17651](https://arxiv.org/abs/2303.17651)) |
| Refinement rounds 3+ | ≈0; quality risk rises | 1 rewrite each | Same; plus [2310.01798](https://arxiv.org/abs/2310.01798) |
| Keep-best archive | Removes all downside risk of the loop | ~0 | Monotonicity by construction |
| Anchor-to-original gates (NLI, sim, entities) | Prevents the StealthRL-style quality collapse | 1 NLI pass | [2305.17493](https://arxiv.org/abs/2305.17493) analogue; report 05 |
| Conformal stopping/rejection calibration | Converts a heuristic τ into a stated guarantee | 500–1,000 labelled calibration docs | [2306.10193](https://arxiv.org/abs/2306.10193), [2110.01052](https://arxiv.org/abs/2110.01052) |
| Escalate/refuse the hard tail (~1–2% of inputs) | The only way past the `1-P(p=0)` ceiling | product decision | §1.2 |
| Frozen reference battery + drift alerting | Preserves all of the above over time | ~$13/day | [sre.google](https://sre.google/workbook/canarying-releases/), [2007.06299](https://arxiv.org/abs/2007.06299) |

---

## 8. A concrete reliability architecture for a 99%+ GPTZero pass rate — and how each assumption fails

### The architecture

**Layer 0 — Admission control.** Classify the input (genre, length, technical density, citation load). Route the known-hard tail (dense technical, <300 words, heavy quotation) to a *different* promise: "borderline, expect manual editing." Predicted `p(x)` comes from a small regressor trained on historical outcomes. **This layer, not sampling, is what makes 99% arithmetically possible.**

**Layer 1 — Decorrelated candidate generation.** N=6 candidates, each with (a) a distinct control-code plan sampled from the genre's human feature distribution (report 04), (b) temperature drawn from 0.7–1.0, (c) rotating base checkpoint across ≥2 model families. Never 6 samples from one model at one temperature with one prompt.

**Layer 2 — Local gate (free).** Hard gates first, as veto: NLI entailment vs. original, E5 cosine ≥ 0.88, entity/number preservation, length ratio ±25%, LanguageTool error rate in the human band. Then score survivors with the 4-family detector ensemble; rank by `max_d score_d` (worst-case), not the mean. Keep-best archive starts here.

**Layer 3 — One targeted refinement round.** For the best candidate, take the top-3 high-impact sentences, rewrite the *paragraphs* containing them with the feature diff supplied as instruction, re-score the whole document, keep only if it improves and gates hold. Second round only if within a small margin of the threshold. Never a third.

**Layer 4 — Live verification cascade.** If `max_d score_d ≤ τ` (τ chosen by Learn-Then-Test on 1,000 labelled docs for a target risk α), send one live detector call. On failure, return to Layer 1 with a *different plan seed* — not the same seed at higher effort. Cap at M=3 live rounds; on exhaustion, return the keep-best candidate with an honest borderline flag.

**Layer 5 — Measurement.** Frozen 200-document reference battery scored daily; stratified 300+ document canary for the published pass rate with Clopper–Pearson bounds; always-valid sequential A/B for pipeline changes; SRE-style 5% canary rollout with auto-rollback on live-pass-rate divergence.

**Expected steady state:** with `p ≈ 0.6`, ρ ≈ 0.15 after Layer 1, `q_τ ≈ 0.85` after Layer 2–3, and M=3 live rounds, the *admitted* population lands around 99.3–99.7% at roughly $0.10–0.15 per document. The blended rate across all traffic is ~98%, and the gap is Layer 0's refusals.

### The assumptions, and how each one fails

1. **"Candidate correlation is manageable (ρ ≈ 0.15)."** Fails if all diversity levers act on surface form while the failure mode is semantic/structural — then every candidate shares it and ρ → 0.5+. *Symptom:* pass rate scales as a power law in N rather than exponentially. *Test:* fit `P(all fail) ∝ N^(-a)` on an N-sweep; if `a < 1`, your diversity is cosmetic. *Fix:* diversify the plan and the base model, not the temperature.
2. **"The local ensemble is a good proxy for GPTZero (`q_τ ≈ 0.85`)."** Fails as soon as GPTZero retrains on humanizer output — which report 03 documents it already does (an "AI (Paraphrased)" class, 93.5% recall on 12 humanizers). *Symptom:* `q_τ` drops while local scores stay excellent. *Test:* the frozen battery's known-humanized arm. *Fix:* Stage-4 self-adversarial discriminator rotated into the ensemble; treat `q_τ` as a monitored SLI, not a constant.
3. **"Best-of-N selection doesn't hurt us."** Fails by Gao's quadratic: at high N you select the candidate that maximally exploits the proxy's blind spot, and gold performance turns over. *Symptom:* pass rate flat or falling as N rises past ~8–16 while local scores keep improving. *Fix:* cap N, add the style/feature-distance term to the ranking so the objective isn't purely detector-shaped.
4. **"The conformal guarantee holds."** Fails on any exchangeability break: detector update, new genre, rewriter change. *Symptom:* battery drift. *Fix:* automatic recalibration trigger; publish the lower confidence bound; consider weighted conformal if you can estimate the shift.
5. **"Refinement doesn't degrade quality."** Fails silently, because embedding similarity hides factual drift and broken collocations (report 03). *Symptom:* LLM-judge win-rate vs. the original input declines while detector scores improve. *Fix:* keep-best + anchor-to-original + LLM-judge as a monitored SLI, not a one-off eval.
6. **"Sentence-level repair helps."** Fails via variance destruction — repairing only the worst sentences flattens the sentence-score profile that burstiness measures. *Symptom:* every sentence score falls, document score rises. *Fix:* paragraph-level edits, full-document re-scoring, and an explicit burstiness gate.
7. **"The hard tail is ~1.5%."** Fails when input mix shifts (a new customer segment writing lab reports). *Symptom:* Layer 0 refusal rate climbs. *Fix:* Layer 0 is a monitored model with its own retraining cadence, not a static rule list.
8. **"Live verification is affordable."** Fails at scale or if the vendor rate-limits/bans automated humanizer traffic — a live business risk, not just a cost one. *Fix:* keep the live call a *sparse calibration* signal (sampled fraction of traffic) rather than a per-document dependency, and make the local ensemble carry the routine load.

The uncomfortable summary: **99% is reachable, but only by conceding two things** — that a measurable slice of inputs must be refused or flagged rather than shipped, and that the last few points come from paying a real detector, not from sampling harder against a surrogate.

---

## Sources

1. Stroebl, Kapoor, Narayanan — *Inference Scaling fLaws / The Limits of Inference Scaling Through Resampling* — https://arxiv.org/abs/2411.17501
2. Brown et al. — *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* — https://arxiv.org/abs/2407.21787
3. Chen et al. — *Evaluating Large Language Models Trained on Code* (pass@k estimator, temperature vs k) — https://arxiv.org/abs/2107.03374 · https://ar5iv.labs.arxiv.org/html/2107.03374
4. Gao, Schulman, Hilton — *Scaling Laws for Reward Model Overoptimization* — https://arxiv.org/abs/2210.10760 · https://ar5iv.labs.arxiv.org/html/2210.10760
5. Beirami et al. — *Theoretical Guarantees on the Best-of-n Alignment Policy* — https://arxiv.org/abs/2401.01879
6. Hughes et al. — *Best-of-N Jailbreaking* — https://arxiv.org/abs/2412.03556
7. Snell et al. — *Scaling LLM Test-Time Compute Optimally…* — https://arxiv.org/abs/2408.03314
8. Zhao, Awasthi, Gollapudi — *Sample, Scrutinize and Scale* — https://arxiv.org/abs/2502.01839
9. Zhang et al. — *Generative Verifiers: Reward Modeling as Next-Token Prediction* — https://arxiv.org/abs/2408.15240
10. Skalse et al. — *Defining and Characterizing Reward Hacking* — https://arxiv.org/abs/2209.13085
11. Rafailov et al. — *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms* — https://arxiv.org/abs/2406.02900
12. Guo et al. — *On Calibration of Modern Neural Networks* (temperature/Platt scaling) — https://arxiv.org/abs/1706.04599
13. Quach et al. — *Conformal Language Modeling* (stopping + rejection rules) — https://arxiv.org/abs/2306.10193
14. Mohri & Hashimoto — *Language Models with Conformal Factuality Guarantees* — https://arxiv.org/abs/2402.10978
15. Angelopoulos et al. — *Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control* — https://arxiv.org/abs/2110.01052
16. Angelopoulos et al. — *Conformal Risk Control* — https://arxiv.org/abs/2208.02814
17. Tibshirani et al. — *Conformal Prediction Under Covariate Shift* — https://arxiv.org/abs/1904.06019
18. Jitkrittum et al. — *When Does Confidence-Based Cascade Deferral Suffice?* — https://arxiv.org/abs/2307.02764
19. Chen, Zaharia, Zou — *FrugalGPT* — https://arxiv.org/abs/2305.05176
20. Yue et al. — *Large Language Model Cascades with Mixture of Thought Representations* — https://arxiv.org/abs/2310.03094
21. Goel et al. — *Great Models Think Alike and this Undermines AI Oversight* (CAPA, correlated errors) — https://arxiv.org/abs/2502.04313
22. Kirk et al. — *Understanding the Effects of RLHF on LLM Generalisation and Diversity* — https://arxiv.org/abs/2310.06452
23. Yue et al. — *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* — https://arxiv.org/abs/2504.13837
24. Zhang et al. — *Verbalized Sampling: Mitigating Mode Collapse* — https://arxiv.org/abs/2510.01171
25. Vijayakumar et al. — *Diverse Beam Search* — https://arxiv.org/abs/1610.02424
26. Holtzman et al. — *The Curious Case of Neural Text Degeneration* (nucleus sampling) — https://arxiv.org/abs/1904.09751
27. *CarBoN: Calibrated Best-of-N Sampling* — https://arxiv.org/abs/2510.15674
28. *Soft Best-of-n Sampling for Model Alignment* — https://arxiv.org/abs/2505.03156
29. *Making, not Taking, the Best of N (Fusion-of-N)* — https://arxiv.org/abs/2510.00931
30. *Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning* — https://arxiv.org/abs/2608.05643
31. Madaan et al. — *Self-Refine: Iterative Refinement with Self-Feedback* — https://arxiv.org/abs/2303.17651 · https://ar5iv.labs.arxiv.org/html/2303.17651
32. Shinn et al. — *Reflexion: Language Agents with Verbal Reinforcement Learning* — https://arxiv.org/abs/2303.11366
33. Gou et al. — *CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing* — https://arxiv.org/abs/2305.11738
34. Huang et al. — *Large Language Models Cannot Self-Correct Reasoning Yet* — https://arxiv.org/abs/2310.01798
35. Stechly, Valmeekam, Kambhampati — *On the Self-Verification Limitations of LLMs* — https://arxiv.org/abs/2402.08115
36. Shumailov et al. — *The Curse of Recursion / Model Collapse* — https://arxiv.org/abs/2305.17493
37. Cheng & Sadasivan et al. — *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text* (NeurIPS 2025) — https://arxiv.org/abs/2506.07001
38. *AuthorMist: Evading AI Text Detectors with RL* — https://arxiv.org/abs/2503.08716
39. *StealthRL: RL Paraphrase Attacks for Multi-Detector Evasion* — https://arxiv.org/abs/2602.08934
40. Lu et al. — *SICO: LLMs Can Be Guided to Evade AI-Generated Text Detection* — https://arxiv.org/abs/2305.10847
41. Dugan et al. — *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors* — https://arxiv.org/abs/2405.07940
42. Weber-Wulff et al. — *Testing of Detection Tools for AI-Generated Text* — https://arxiv.org/abs/2306.15666
43. *Rethinking AI-Generated Text Detection: A Strong Baseline and the Distribution-Shift Problem That Remains* — https://arxiv.org/abs/2607.03680
44. Tripathi et al. — *DoGEN: Domain Gating Ensemble Networks for AI-Generated Text Detection* — https://arxiv.org/abs/2505.13855
45. Dubois, Yvon, Piantanida — *MOSAIC: Multiple Observers Spotting AI Content* — https://arxiv.org/abs/2409.07615
46. Liu et al. — *Delving into Transferable Adversarial Examples and Black-box Attacks* (ensemble transfer, Clarifai) — https://arxiv.org/abs/1611.02770
47. Xie et al. — *Improving Transferability of Adversarial Examples with Input Diversity* — https://arxiv.org/abs/1803.06978
48. Liu et al. — *Min-Max Optimization without Gradients* (ensemble attack, worst-case weighting) — https://arxiv.org/abs/1909.13806
49. Jin et al. — *Is BERT Really Robust? (TextFooler word-importance ranking)* — https://arxiv.org/abs/1907.11932
50. Dale et al. — *Text Detoxification using Large Pre-trained Neural Models (CondBERT span replacement)* — https://arxiv.org/abs/2109.08914
51. Johari et al. — *Peeking at A/B Tests: Always Valid Inference* — https://arxiv.org/abs/1512.04922
52. Klaise et al. — *Monitoring and Explainability of Models in Production* — https://arxiv.org/abs/2007.06299
53. *Adversarial Validation Approach to Concept Drift at Uber* — https://arxiv.org/abs/2004.03045
54. Google SRE Workbook — *Canarying Releases* — https://sre.google/workbook/canarying-releases/
55. NIST/SEMATECH e-Handbook — *Confidence Intervals for Proportions* — https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
56. *Rule of Three (statistics)* — https://en.wikipedia.org/wiki/Rule_of_three_(statistics)
57. GPTZero — *Technology / model, sentence classifier, API fields* — https://gptzero.me/technology
58. GPTZero Support — *Interpreting Advanced Sentence Scanning* — https://support.gptzero.me/articles/7549392421-how-do-i-interpret-results-from-gptzero-s-advanced-sentence-scanning
59. Originality.ai — *Pricing (credits, 1 credit = 100 words)* — https://originality.ai/pricing
60. Undetectable.ai — *Product description* — https://undetectable.ai/
