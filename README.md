# Main Street Marketplace Toolkit

**Open, non-proprietary marketplace intelligence for U.S. small businesses.**

`msmt` is a Python toolkit that translates the kind of analytics enterprise
marketplaces use internally — listing integrity, supply resilience, demand
forecasting with practitioner guardrails — into transparent, auditable
modules that an SBDC counselor, a state commerce program analyst, a niche
marketplace operator, or an SMB seller can run on their own laptop.

It is a companion to the **Main Street Marketplace** article series, a
five-part, plain-language walk-through of the same ideas:

- Part 1 — *(coming soon)*
- Part 2 — *(coming soon)*
- Part 3 — *(coming soon)*
- Part 4 — *(coming soon)*
- Part 5 — *(coming soon)*

There is no commercial version of this toolkit and no paid tier. It is MIT
licensed, free forever, and intentionally built from public methods and
synthetic data so anyone can read, run, and adapt it without asking
permission.

## Who this is for

- **SBDC counselors** advising small marketplace sellers who want a
  defensible, explainable view of demand and inventory risk.
- **State and regional commerce program staff** evaluating where small
  sellers are most exposed to marketplace volatility.
- **Niche marketplace operators** (regional, industry-vertical, or co-op
  marketplaces) who need ideas that scale down to a few hundred SKUs.
- **SMB sellers** on Amazon, Walmart, eBay, Shopify, Etsy, or any
  comparable platform who want to reason about their own data without
  handing it to a black-box vendor.
- **Policy researchers** studying small-business resilience in
  digital commerce.

## The three pillars

The toolkit is organized around three problems every marketplace seller
runs into, regardless of size:

1. **Marketplace integrity** (`msmt.integrity`) — Spot listing health,
   review-pattern, and content-quality issues that put a seller at risk
   of suppression or buyer-trust erosion. *(Built later in the series.)*
2. **Supply resilience** (`msmt.resilience`) — Translate lead times,
   stock-on-hand, and demand variability into stockout risk and reorder
   guardrails a non-specialist can actually act on. *(Built later in the
   series.)*
3. **Forecasting with guardrails** (`msmt.forecasting`) — Forecast
   short-horizon demand for the five archetypes small sellers actually
   see, with explicit uncertainty and "don't trust this forecast"
   warnings when history is too short or too lumpy. *(Built later in the
   series.)*

Shared utilities live under `msmt.data` (synthetic data generation,
shipped today) and `msmt.reports` (notebook and markdown reporting
helpers, added when the modules above land).

## Status

This is **Day 1** of a four-day initial build. What ships in this
release:

- `msmt.data.synthetic` — a reproducible synthetic seller data
  generator covering all five demand archetypes the toolkit targets.

The integrity, resilience, and forecasting modules will follow in later
sessions and will all consume the same synthetic data interface, so
examples and tests stay end-to-end runnable without anyone needing to
share real seller data.

## Quickstart

Install from source (a PyPI release will come once the API stabilizes):

```bash
git clone https://github.com/ayushtripathi955/main-street-marketplace-toolkit.git
cd main-street-marketplace-toolkit
pip install -e .
```

Generate a synthetic multi-SKU dataset:

```python
from msmt.data import generate_seller_data

df = generate_seller_data(n_skus=50, n_days=365, seed=42)
print(df.head())
print(df["pattern"].value_counts())
```

You will get a tidy, daily DataFrame with the columns `date`, `sku_id`,
`units_sold`, `listing_price`, `stock_on_hand`, `lead_time_days`,
`category`, and `pattern`. Feed any column into your own analysis — the
rest of the toolkit will use this exact schema.

Need a single SKU of one specific archetype (handy for unit tests or
worked examples)?

```python
from msmt.data import (
    generate_smooth,
    generate_weekly_seasonal,
    generate_holiday_spike,
    generate_intermittent,
    generate_new_sku,
)

steady = generate_smooth(n_days=180, seed=0)
seasonal = generate_weekly_seasonal(n_days=180, seed=0)
holiday = generate_holiday_spike(n_days=365, seed=0, end_date="2025-12-31")
lumpy = generate_intermittent(n_days=365, seed=0)
brand_new = generate_new_sku(n_days=180, history_days=30, seed=0)
```

Every generator accepts a `seed`, so the same arguments always return
the same DataFrame.

## A note on the data

**All data produced by this toolkit is synthetic.** It is not derived
from, sampled from, or modeled on any proprietary or real-world seller
dataset. The five demand archetypes are well-known patterns from the
public operations-research and demand-forecasting literature; the
generators here are simple, transparent implementations meant to make
the rest of the toolkit demonstrable end-to-end without anyone needing
to share their real numbers.

If you want to run `msmt` against your own data, just match the
schema — the column names listed in the quickstart are all the toolkit
expects.

## License

[MIT](LICENSE). Free to use, modify, and redistribute. No attribution
required, though a link back to the article series is appreciated.

## Author

**Ayush Tripathi** — data analytics and marketplace strategy
practitioner based in San Francisco. The toolkit and accompanying
article series were built to give Main Street sellers and the people
who advise them the same kind of marketplace intelligence enterprise
sellers take for granted, in a form anyone can read, run, and trust.
