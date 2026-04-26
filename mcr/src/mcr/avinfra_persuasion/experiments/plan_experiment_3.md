# Plan for `experiment_3`

## Goal

Move from a finite prior over explicit scenarios to a continuous prior with
compact support over the primitive network measures, while keeping the routing
side readable and close to the current code.

The first implementation target is:

- continuous prior over network metrics,
- public masked signaling policy,
- truthful **quantized** disclosure on instrumented components,
- receiver-side Bayesian update via likelihood reweighting,
- route choice by rescoring a cached prior route set.


## Core design decisions

### 1. This is a different inference regime

For finite priors, receiver updating is exact support filtering:

- keep scenarios consistent with the signal,
- renormalize.

For continuous priors, exact equality on a realized continuous message is a
zero-measure event. So `experiment_3` must use a likelihood-based update.


### 2. Continuous uncertainty lives in a generative prior

Do **not** model this as “just sample many scenarios” with no structure.

We want a real prior object that:

- knows the support of each uncertain metric,
- can sample scenarios,
- and can support receiver-side Bayesian updating through a likelihood model.

Initial compact supports:

- `travel_time`: per arc, `Uniform(nominal_tt[arc], 10 * nominal_tt[arc])`
- `hazard`: per arc, `Uniform(0, 4)`
- `discomfort`: per arc, `Uniform(0, 4)`
- `cost`: per arc, `Uniform(0, 5)`
- `policing`: per node, `Bernoulli(p_node)`

Two metrics remain non-primitive:

- `emissions`: derived from the realized network state
- `left_turns`: structural, path-dependent only


### 3. Public signaling policy matters, but only through the right term

The Bayesian update is

\[
P(\theta \mid m) \propto P(m \mid \theta)\, P(\theta)
\]

where:

- `θ` is the world state / scenario,
- `m` is the realized message.

Because the signaling policy is public, the receiver should use it in the
update.

However, for the **current** mask policy class, the metric-reveal mask is
state-independent:

- each metric is revealed with some Bernoulli probability,
- that probability does not depend on the realized world.

So under this policy class:

- the mask itself is not informative about the state,
- the observed disclosed values are informative.

This distinction matters for the rollout:

- **Phase 1** keeps the mask state-independent and uses truthful quantized
  disclosure,
- **Phase 2** may introduce state-dependent message rules, where the policy
  itself becomes informative.


### 4. Start with quantized truthful disclosure

We first want a truthful observation channel, not a fully strategic continuous
message policy.

Two candidate truthful channels were discussed:

1. quantized truthful disclosure,
2. noisy truthful disclosure.

We choose **quantized truthful disclosure first** because it:

- gives a finite message space,
- is easier to inspect and test,
- avoids immediate particle collapse from exact real-valued observations,
- keeps the update logic simple and explicit.

Example:

- instead of revealing `travel_time = 2.137`,
- reveal `travel_time in [2.0, 2.5)`.


## Posterior representation

We should not try to represent the continuous posterior symbolically.

Use a **particle belief**:

- particles are sampled `Scenario`s,
- weights approximate the posterior,
- updates happen by likelihood reweighting,
- optional resampling can be added later if needed.

This gives a clean bridge to the current routing code because routing already
works from sampled scenarios and expected metric totals.


## Receiver update rule

### Finite prior case

Keep the current exact update for `FinitePrior`.

### Continuous / particle case

For a weighted particle belief, update as:

\[
w_i' \propto w_i \cdot P(m \mid \theta_i)
\]

For quantized truthful disclosure:

- if a particle’s realized value falls inside the revealed bin, likelihood is
  nonzero,
- otherwise likelihood is zero.

For `policing`:

- likelihood is Bernoulli-consistent with the observed value.

In words:

- hidden metrics do not contribute to the likelihood,
- revealed metrics contribute through bin membership on instrumented arcs/nodes.


## Routing side

The route-generation pattern from `PriorRouteChoiceReceiver` should stay.

That means:

- generate the prior-efficient route set once,
- update beliefs after signals,
- rescore the cached route set under the updated belief,
- choose a maximal route under the receiver preference.

This already matches the current refactor:

- `PriorRouteChoiceReceiver`
- `RoutingSolution.rescore(...)`

For `experiment_3`, the only required extension is:

- the rescoring logic must work from **weighted scenarios**, not just finite
  support probabilities.

That keeps the continuous case aligned with the finite case.


## Phase 1: inference-correct continuous toy

This is the first implementation target.

### Scope

- continuous prior with compact support,
- particle posterior approximation,
- public masked signaling policy,
- truthful quantized disclosure on instrumented components,
- cached-route rescoring via `PriorRouteChoiceReceiver`.

### New abstractions

#### `ContinuousPrior`

A real prior object that:

- stores the compact-support specifications,
- samples `Scenario`s,
- can initialize a particle approximation.

Possible shape:

- one prior object over the whole world,
- internally stores per-metric, per-arc / per-node distribution rules.

#### `ParticleBelief`

A weighted belief object with:

- `particles: tuple[Scenario, ...]`
- `weights: tuple[float, ...]`

and methods like:

- `sample(...)`
- `normalized()`
- `effective_sample_size()`
- optional `resample(...)`

#### `QuantizedSignal`

We can either:

1. keep using `Signal.value` and store bins as payload values, or
2. introduce a dedicated signal type for quantized observations.

The minimal first step is:

- keep `Signal`,
- let `value` store bin descriptors for revealed components.

### Receiver update in phase 1

- `FinitePrior` -> current exact filtering
- `ContinuousPrior` -> initialize particles, then update
- `ParticleBelief` -> reweight directly

### Sender behavior in phase 1

Keep the sender policy simple:

- choose which metrics are disclosed,
- materialize truthful quantized observations on instrumented arcs/nodes.

No strategic distortion yet.


## Phase 2: strategic continuous signaling

Only after phase 1 is stable.

Possible extensions:

- sender chooses disclosure precision,
- sender chooses bin boundaries,
- sender chooses state-dependent message distributions,
- sender reveals coarse vs fine information strategically.

At that point the public signaling policy enters the receiver update directly as
part of `P(m | θ)`, not just through the truthful observation channel.


## File-level implementation plan

### `datastructures.py`

Add:

- `ContinuousPrior`
- `ParticleBelief`
- compact-support distribution specs for arc/node metrics

Keep:

- `Scenario` as the realized bundle
- `emissions` derived
- `left_turns` structural

Potential helper:

- `weighted_scenarios()` interface or equivalent helper so finite and particle
  beliefs can be treated uniformly during rescoring.


### `bp/signals.py`

Add the phase-1 signal representation for quantized truthful disclosure.

Minimal approach:

- keep `MaskSignalPolicy` for metric selection,
- store quantized observations in `Signal.value`.

If that becomes too implicit, introduce a dedicated quantized signal class.


### `bp/senders.py`

Add sender-side materialization for quantized truthful disclosure:

- sample / receive the metric mask,
- on instrumented components, map realized values to bins,
- store those bins in the signal payload.

Keep the current distinction:

- arc metrics on instrumented arcs,
- node metrics on nodes incident to instrumented arcs.


### `bp/receivers.py`

Extend `update_internal_belief(...)`:

- finite prior -> exact filter,
- particle belief -> likelihood reweight,
- continuous prior -> initialize particles then update.

Keep `PriorRouteChoiceReceiver` as the main route-choice mechanism.


### `opt.py`

No new optimization logic should be needed for phase 1.

The main requirement is that rescoring from weighted scenarios remains clean.


### `experiments/experiment_3.py`

Build the first continuous toy around:

- one continuous prior,
- one public mask policy,
- one truthful quantized disclosure channel,
- one or more prior-route receivers.

Start with a small toy network and small particle count so behavior is easy to
inspect.


## Testing plan

### Continuous prior

- sampled travel times stay within `[nominal, 10 * nominal]`
- sampled hazards stay in `[0, 4]`
- sampled discomfort stays in `[0, 4]`
- sampled cost stays in `[0, 5]`
- policing samples are binary

### Quantized signaling

- revealed values map to the correct bins
- hidden metrics produce no payload
- only instrumented components are disclosed

### Receiver update

- particles inconsistent with a revealed bin get zero likelihood
- posterior weights renormalize correctly
- hidden metrics leave weights unchanged
- finite-prior update path still works

### Routing

- cached prior routes are reused
- rescoring under weighted particles changes chosen routes when the posterior
  changes


## Open decisions before coding

### 1. Bin design

We still need to choose how bins are defined:

- fixed global bins per metric,
- arc-specific bins,
- or bins derived from nominal values.

Recommended first step:

- `travel_time`: bins scaled from nominal travel time,
- `hazard`, `discomfort`, `cost`: fixed bins on their compact support,
- `policing`: binary, so no quantization needed.

### 2. Particle count

We need a default particle count for phase 1.

Recommendation:

- start small enough to inspect behavior,
- large enough to avoid obviously unstable routing choices.

### 3. Policing prior

We need to set `p_node`.

Recommended first step:

- use nominal policing as the prior mean per node.


## Recommended next implementation step

Implement **phase 1 only**:

1. `ContinuousPrior`
2. `ParticleBelief`
3. truthful quantized masked disclosure
4. likelihood-based receiver update
5. `experiment_3` on top of `PriorRouteChoiceReceiver`

Do **not** implement strategic continuous messaging until the inference path is
working and inspectable.
