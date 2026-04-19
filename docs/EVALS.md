# Evals: the differentiator

**If you don't have evals, you don't have an agent product. You have a
demo.** This is the single biggest reason agent projects stall: every
prompt tweak is a coin flip without a regression harness to catch the
damage.

Agent Kit gives you a batteries-included eval loop:

- Golden datasets in JSONL (diffable, reviewable in a PR).
- Four judges out of the box — string match + three LLM judges.
- Parallel runner with budgets.
- Regression compare against a baseline run.
- Markdown + HTML reports for sharing, console table for CI.

## The 30-second loop

```bash
# 1. Write 5-20 example cases to a JSONL file.
#    See examples/eval_suites/basic.jsonl for the format.
cat > suites/my_suite.jsonl <<'EOF'
{"id": "q1", "input": "What is Agent Kit?", "expected": "starter kit"}
{"id": "q2", "input": "What language?", "expected": "Python"}
EOF

# 2. Run them.
uv run agent-kit-evals run suites/my_suite.jsonl

# 3. Read the output.
open eval_runs/my_suite.html
```

The HTML report shows each case, judge scores, the raw output, and a
colour-coded pass/fail badge. Ship it to a Notion page. Attach it to the PR.

## The JSONL schema

```jsonc
{
  "id": "stable-id",          // required, used for regression diffs
  "input": "user prompt",     // required
  "expected": "hint-string",  // optional, used by judges
  "tags": ["math", "smoke"],  // optional, used for filtering
  "metadata": {"note": "…"}   // optional, free-form
}
```

- Keep `id`s stable. Regression compare joins on it.
- `expected` doesn't have to be a full answer. For LLM judges, treat it as
  a *rubric hint*: "the output should mention Paris and be under 50 words".
- Use `tags` to run subsets (`--tags math`) and aggregate per-tag scores.

## Judges

A judge takes `(case, output)` and returns a `JudgeResult(score, passed,
rationale)`. Agent Kit ships four.

| Judge                | Kind        | What it's for                                 |
|----------------------|-------------|-----------------------------------------------|
| `StringMatchJudge`   | Deterministic | Smoke checks — fast, free, unambiguous.     |
| `CorrectnessJudge`   | LLM-as-judge | "Does the answer match `expected`?"          |
| `HelpfulnessJudge`   | LLM-as-judge | "Is the answer actually useful?"             |
| `SafetyJudge`        | LLM-as-judge | "Would this output hurt someone / violate policy?" |

LLM judges take a `Provider`. By convention we use a strong model
(`claude-sonnet-4-6` or `gpt-5`) as judge even when the agent under test
uses a cheaper one — **judge quality dominates eval quality**.

### Writing a custom judge

```python
from agent_kit.evals.judges import Judge, JudgeResult
from agent_kit.evals.datasets import EvalCase

class MaxLatencyJudge(Judge):
    name = "max_latency"
    threshold = 1.0

    def __init__(self, max_ms: float):
        self.max_ms = max_ms

    async def score(self, case: EvalCase, output: str) -> JudgeResult:
        # latency lives in metadata produced by the harness? no — add your own
        # signal by wrapping the factory:
        ...
```

Mix multiple judges and the harness will average their scores; pass/fail
is AND across all.

## Regression testing

Pin a baseline run (check it into git, or S3):

```bash
uv run agent-kit-evals run suites/my_suite.jsonl \
    --output-dir baselines
# → baselines/my_suite.json
```

On every PR:

```bash
uv run agent-kit-evals run suites/my_suite.jsonl \
    --baseline baselines/my_suite.json
```

The runner prints a regression table with per-metric deltas and a list of
newly-failing case IDs. Fail CI on regressions:

```bash
uv run agent-kit-evals compare baselines/my_suite.json \
    eval_runs/my_suite.json > diff.json
python -c "import json,sys;d=json.load(open('diff.json'));\
sys.exit(1 if d['regressed_cases'] else 0)"
```

## Programmatic harness

The CLI is a thin wrapper. For fine control:

```python
from agent_kit.evals.harness import EvalHarness, default_judges
from agent_kit.evals.datasets import load_jsonl
from agent_kit import Agent, Settings

async def factory(case):
    return Agent(settings=Settings(), system="You are a senior developer.")

harness = EvalHarness(
    agent_factory=factory,
    judges=default_judges(),
    concurrency=8,
)
run = await harness.run(load_jsonl("suites/my_suite.jsonl"))
print(run.metrics)  # RunMetrics(pass_rate=..., cost_usd=..., ...)
run.save("eval_runs/my_suite.json")
```

## Cost awareness

- Every case records its dollar cost. The run summary shows the total.
- LLM judges consume tokens too. Their cost is tracked under the judge
  model's entry in the pricing table.
- A 50-case suite with 3 LLM judges typically costs $0.10 - $0.50
  depending on model choice.

## Three suites we ship

- `examples/eval_suites/basic.jsonl` — smoke tests: arithmetic, trivia,
  formatting. Use for "did I break something obvious?"
- `examples/eval_suites/coding.jsonl` — tiny programming prompts. Stress
  tests your agent's structured output.
- `examples/eval_suites/safety.jsonl` — jailbreak / refusal / PII. If you
  regress on any of these, it's a big deal.

## Workflow we recommend

1. **Before writing a new prompt**, add the happy-path and 2 tricky cases
   to your suite.
2. **Every prompt change** → local run → check deltas → commit.
3. **CI on every PR** → run the suite, block merge on regression.
4. **Weekly**, prune cases that are always green (they add cost without
   signal). Keep the suite roughly 50 cases — big enough to notice,
   small enough to run in under 5 minutes.

Once you have this loop, you stop worrying. Ship faster.
