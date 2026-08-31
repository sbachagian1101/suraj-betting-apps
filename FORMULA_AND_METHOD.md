# Formula and modelling method

## Specialist probabilities

Each specialist model creates a runner score and converts it into a race-level probability through softmax:

\[
P_{im}=\frac{\exp(s_{im}/T_m)}{\sum_j\exp(s_{jm}/T_m)}
\]

where `i` is the runner, `m` is the specialist model, and `T` controls probability sharpness.

## Baseline logarithmic pool

For a normal flat race, the starting combination is approximately:

\[
30\%\text{ Bet365 Analyst}
+34\%\text{ Independent Form}
+22\%\text{ Suitability/Fitness}
+14\%\text{ Pace/Draw}
\]

For hurdles and steeplechases, relevant obstacle form, stamina and completion evidence receive additional influence. Debutant-heavy races give more weight to the source and low-data information.

The combination is a logarithmic pool:

\[
\log S_i=\sum_m w_m\log(P_{im})
\]

The baseline probability is obtained after normalization across the active field.

## Form quality

A historical finishing percentile is:

\[
q=1-\frac{\text{finish}-1}{\text{field size}-1}
\]

Winning and beaten margins are transformed into bounded quality values. Historical performances receive recency weights:

\[
w_r=\exp(-\text{days ago}/85)
\]

Distance similarity is:

\[
d_r=\exp\left(-\frac{|D_r-D_0|}{\max(180,0.22D_0)}\right)
\]

Class-adjusted form combines finishing percentile, margin quality, race strength, recency and distance relevance.

## Fitness

Days since the latest run are modelled with a broad optimum rather than a single hard threshold. First-up, second-up, third-up and later-preparation wording adjusts that value. First starters receive neutral ability rather than a zero score, but their confidence is reduced.

## Pairwise learning

For each known result, pairwise winner-loser feature differences are created:

\[
x_{ab}=x_a-x_b
\]

The learned probability that runner `a` beats runner `b` is:

\[
P(a>b)=\frac{1}{1+\exp(-\beta^Tx_{ab})}
\]

The coefficients are fitted with L2 regularization. Partial results and dead heats are preserved.

## Final trained score

The learned score is blended with the transparent baseline:

\[
F_i=(1-\alpha)B_i+\alpha L_i
\]

where `alpha` grows gradually with accumulated races and is capped at 52%. The bundled 35-race state begins at approximately 21.1% learned influence.

## Complete finishing-position estimates

The final win distribution is used in deterministic-seeded Plackett-Luce simulations. These produce:

- Top-three probability
- Expected finishing rank
- Full predicted finishing order

## Confidence

The 0-9 confidence score combines:

- Specialist-model agreement
- Form-data coverage
- First-starter uncertainty
- Separation from the field median
- Separation between the leading probabilities

## Exclusions

The feature schema excludes current and historical betting prices. Actual results enter only after a pre-race prediction snapshot has been frozen.
