# Match Insight

Paste the last five FootyStats match pages for each of two teams. The app reads
them, builds each side's rates, and runs **thirteen prediction methods** over
those rates to produce 1X2, BTTS, over/under 2.5, corners over/under 8.5 and a
full correct-score grid.

## How it works

**Parse.** A FootyStats page is mostly noise — league tables, top scorers, the
footer in thirty languages. Four blocks carry everything worth having: the date
line, the `TeamA vs TeamB` line, `Final Results` with the score, and the `Data`
block with possession, shots, cards, corners, fouls, offsides and xG. Pages
pasted twice — the head-to-head appears in both teams' sets — are
de-duplicated.

**Weight.** Each match carries `recency × importance`. Recency is exponential
decay on days before the most recent match, so a fixture from the previous
season counts for a fraction of last weekend's. Importance comes from the
competition named on the page: a **friendly** is evidence about a squad rather
than a team and is discounted hard, a cup tie sits between. Every per-match
weight is shown on the Parsed data tab, and both dials are adjustable.

**Index.** *Attack strength* is chances created against the sample average;
*defence strength* is chances allowed, inverted, so above 1 is good for both.
*Attack weakness* is the share of xG not converted; *defence weakness* the
share conceded above xGA. A team can be strong and wasteful at once, and the
two indices say so separately.

**Predict.** Thirteen methods, then a weighted ensemble.

| | method | what makes it different |
|---|---|---|
| 1 | Poisson (goals) | independent Poisson on goal rates |
| 2 | Poisson (xG) | the same, driven by xG |
| 3 | Dixon–Coles | Poisson plus the low-score dependence correction |
| 4 | Bivariate Poisson | a shared component correlates the two scores |
| 5 | Negative binomial | allows scores more spread out than Poisson |
| 6 | Skellam | exact distribution of the goal difference |
| 7 | Monte Carlo | simulates with the *rates themselves* uncertain |
| 8 | Shots × conversion | goals from shot volume and finishing rate |
| 9 | Empirical resample | resamples the scorelines actually produced |
| 10 | Bradley–Terry | strength fitted to results, Davidson draw term |
| 11 | Elo | sequential ratings over the parsed matches |
| 12 | Form logistic | ordered logit on the points-per-game gap |
| 13 | Weakness-adjusted | rates nudged by the finishing and keeping gaps |

## Shrinkage, and why the numbers look tame

Five matches is about ten numbers. An unshrunk five-match scoring rate is a
**worse** forecast than simply using the average, because nearly all of its
movement is noise — a single 4-2 shifts it by 0.4 goals a game. Every rate is
pulled toward the sample mean by `n/(n+4)`, so a team's own record carries a
little over half the weight. The probabilities therefore sit closer to the
middle than a five-match record might suggest. That is the correct response to
a small sample, not timidity.

## What thirteen methods is not

**It is not thirteen opinions.** Ten of them are fed by the same two scoring
rates and differ only in the count distribution wrapped around them. They will
agree, and their agreement is not evidence of anything. The app reports the
spread across methods and says plainly, on the page, that it is not a
confidence interval.

The real uncertainty is in the rates, and on five matches a side it is large.
There is also **no market price** in this input and **no measured track
record** — nothing here has been validated out of sample, because the input is
one fixture pasted by hand. Treat the output as a structured reading of the
form you pasted, not as a tested forecast.

## Traps found while building this

**Club names contain digits.** Rejecting any candidate team name with a digit
in it — a reasonable-looking way to avoid matching scores and dates — silently
dropped *First Vienna FC 1894*, and with it one of Sturm Graz II's five
matches. The test is now that a name must not be *mostly* digits. Schalke 04
and 1860 München would have gone the same way.

**A transposed table can be untypeable.** Showing the indices with a team per
column puts the text row (form) in the same column as the numeric ones. Arrow
cannot type that, and Streamlit ships dataframes to the browser as Arrow, so
the table fails to render at all. Values are formatted to strings before the
transpose.

## Files

| | |
|---|---|
| `parser.py` | pulls matches out of pasted FootyStats pages |
| `metrics.py` | weights, rates, shrinkage, strength and weakness indices |
| `methods.py` | the thirteen methods, the ensemble, the corners market |
| `app.py` | the Streamlit interface |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
python test_model.py && python test_app.py
```

181 checks — 148 on the parser, metrics and methods, 33 driving the app. The
app checks press the Predict button and read the rendered grid, not just the
landing page: a Streamlit app with a fatal error deeper in the script still
serves HTTP 200 and still renders its first tab.

---

*Probabilistic decision support, not a guaranteed outcome. Gamble responsibly —
Gambling Help 1800 858 858, [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au).*
