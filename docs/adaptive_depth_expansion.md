## Motivation
**When should the system go deeper?**
Depth is not always beneficial. Unnecessary recursion wastes compute and can overfit sub-expressions to noise. The system should expand depth **adaptively**, triggered by evidence that the current search scale is exhausted.

---
## Core Mechanism
The same multi-agent architecture is applied **recursively at multiple scales**:
```
Scale 0 (global): Multi-agent SR finds y = f(x_1, x_2, ..., x_n)
  |
  +-- "f contains a sub-expression g(x_1, x_2) that resists simplification"
  |
  Scale 1 (sub-expression): Multi-agent SR finds g(x_1, x_2) = ?
    |
    +-- "g contains a nonlinear kernel h(x_1) that needs its own search"
    |
    Scale 2 (kernel): Multi-agent SR finds h(x_1) = ?
      |
      +-- resolves to h(x_1) = log(1 + exp(x_1))  [softplus]

Unwind:
  g(x_1, x_2) = a * log(1 + exp(x_1)) + b * x_2
  f = ... + c * g(x_1, x_2) + ...
```

At each scale, the *same* agent architecture is instantiated. Sub-expression agents report back to their parent agent when they have found a satisfactory form.

---
## Stagnation-Triggered Depth Expansion
Rather than expanding depth with a fixed probability or at every compound term, depth expansion is **gated by flat search stagnation**.
### Mechanism
1. Track the improvement rate $\Delta S$ over the last $k$ outer-loop iterations at the current scale.
2. When $\Delta S$ drops below a threshold $\epsilon$ (i.e., flat search has stalled), allow the proposer agent to:
   - Identify which sub-expression in the current structure contributes most to the residual error.
   - Wrap that sub-expression in a new compound term.
   - Spawn a sub-agent to resolve the deeper decomposition.
### Why stagnation-triggered is better than fixed probability

| Approach                 | Problem                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Always recurse           | Wastes compute on sub-expressions that are already well-resolved                                                              |
| Fixed probability        | No theoretical motivation for the probability value; recurses at random times                                                 |
| **Stagnation-triggered** | Theoretically motivated: switches from exploitation to structural exploration when the current abstraction level is exhausted |

This is analogous to **adaptive operator selection** in evolutionary computation: the search strategy adapts based on the fitness landscape, not a fixed schedule.
### Difficulty heuristics for sub-expressions
When stagnation triggers depth expansion, the system selects *which* sub-expression to decompose using:
- **Depth of nesting** in the current expression
- **Number of variables involved** in the sub-expression
- **Score improvement** when the sub-expression was last modified (low improvement = high resistance = good candidate for decomposition)
- **Residual correlation** with the sub-expression's variables

---
## Why Agents, Not Ordinary Recursive Rewrite?

A mechanical recursive rewrite (e.g., "if node depth > $k$, apply grammar rules") can also decompose expressions hierarchically. The critical question is: **what do agents add?**
### What agents provide over recursive rewrite

1. **Context-dependent decomposition.** An agent can look at the residual and identify *which* sub-expression is actually the bottleneck, instead of blindly recursing on every node. A rewrite rule doesn't condition on fitness signals.

2. **Adaptive strategy selection.** At each scale, the sub-problem may look very different. An agent can choose exploration vs. exploitation, or shift combinator preferences based on the local error pattern. Recursive rewrite applies the same grammar uniformly.

3. **Semantic context passing.** Agents can propagate soft constraints downward: "this sub-expression should be bounded," "monotonically increasing in $x_1$," or "captures diminishing returns." These come from parent-level residual analysis and domain knowledge. Recursive rewrite only passes the symbolic expression.

4. **Non-trivial termination.** An agent can judge "good enough" using the fitness landscape and complexity tradeoffs. Recursive rewrite needs hard-coded stopping rules that can't adapt to the problem.

### Required ablation

To make the agent contribution convincing, we need an empirical ablation:

| Variant | Description |
|---------|-------------|
| **Flat (baseline)** | Single-scale outer loop, no depth expansion |
| **Recursive rewrite** | Mechanical depth expansion using DSL grammar rules, triggered by same stagnation signal |
| **Adaptive agents** | Agent-based depth expansion with context-dependent decomposition and strategy selection |

**Hypothesis:** Adaptive agents outperform mechanical recursive rewrite specifically because decomposition decisions require context (residual patterns, error analysis, domain semantics) that fixed rules cannot capture.

If agents don't beat recursive rewrite empirically, the contribution is weaker and the simpler approach should be preferred.

---

## Application to Structure Descent

### Why this project is a good fit
- The utility $U(k) = \sum_j \theta_j \cdot \phi_j(k)$ is a linear combination of DSL terms, but the complexity lives in the combinators (`interaction`, `ratio`, `split_by`, `power`, `log_transform`, etc.).
- The current outer loop relies on a single LLM reading diagnostics and proposing one change at a time. It struggles with deep, nested, or highly nonlinear compositions.
- The hierarchical weight decomposition (global + category + customer) already has a multi-scale flavor -- the recursive idea extends this naturally to the *structure* level.

### Concrete example

```
Global (Scale 0):
  Top agents propose overall terms: ["routine", "affinity", "interaction(routine, popularity)"]
  Flat search stagnates after 5 iterations. Stagnation detector fires.
  Residual analysis identifies interaction(routine, popularity) as the bottleneck.

Sub-expression (Scale 1):
  Spawn agents to optimize the interaction part.
  Agents discover: interaction(routine, log_transform(popularity))

Kernel (Scale 2):
  If needed, another spawn optimizes a nonlinear kernel inside.
  E.g., turning a raw count into softplus or power(., 0.5)

Unwind:
  Sub-agents report best form + weight contribution back to parent.
  Parent inserts it as a new primitive and continues.
```

### Benefits
- **Scalability:** Discovering deep interactions like `split_by(log_transform(affinity), price_sensitivity > threshold)` becomes feasible because each layer solves a smaller problem.
- **Interpretability:** Sub-expressions come with their own diagnostics and justifications ("this kernel captures diminishing returns on repeat purchases").
- **Better exploration:** The agent at each level can focus on behavioral intuition relevant to that scale (global agents think about overall utility, kernel agents think about saturation effects).
- **Synergy with hierarchical weights:** Discovered sub-expressions can inherit or benefit from the global + category + customer weight deviations automatically.

---
## Implementation Notes
- **Start small:** Add recursion only for combinators involving 2+ primitives or depth > 2.
- **Max depth limit:** Cap at 3-4 levels to prevent infinite recursion.
- **Sub-expression as new primitive:** When a sub-agent reports back, treat the discovered sub-expression as a new DSL primitive for higher levels. This is how InceptionSR and similar hierarchical SR methods work.
- **Diagnostics propagation:** Include "why this sub-form helps" in the report to the parent LLM, so the parent can make informed decisions about further structural changes.
- See `recursive_multi_agent.md` for the detailed code-level implementation plan.

---
## Gap in the Literature
- **Single-agent LLM-SR** with iterative prompting (LLM-SR, ICSR) -- one LLM proposes one complete expression per round.
- **LLM-as-mutation-operator** in evolution (FunSearch, AlphaEvolve, ELM, LaSR) -- one population of complete expressions.
- **Multi-agent debate/collaboration** for reasoning (Du et al., ReConcile, Corex) -- agents argue about the same answer, not sub-components.

