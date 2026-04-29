# Contributing

Thanks for your interest in the Main Street Marketplace Toolkit. This
project exists to give SBDC counselors, state commerce program staff,
niche marketplace operators, and SMB sellers a transparent,
non-proprietary view of marketplace intelligence — so contributions
that make the toolkit easier to read, run, or adapt are especially
welcome.

## Setting up

```bash
git clone https://github.com/ayushtripathi955/main-street-marketplace-toolkit.git
cd main-street-marketplace-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[forecasting,notebooks,app,test]"
```

If `prophet` fails to build on your platform, the install will surface
a wheel error. Either skip the `forecasting` extra
(`pip install -e ".[notebooks,app,test]"`) — `msmt.forecasting`'s
Prophet wrapper falls back to a pure-numpy additive model — or follow
the platform-specific Prophet install instructions at
<https://facebook.github.io/prophet/docs/installation.html>.

## Running the test suite

```bash
.venv/bin/python -m pytest tests/ -q
```

The full suite runs in under two seconds. Every PR should keep the
test count steady or growing — tests double as living documentation
of the library's invariants.

## Running the app locally

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

If you change the app, please verify all four pages still render
cleanly via the headless `AppTest` API:

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
at.run()
assert not at.exception
```

## Project conventions

A few things the codebase tries to be consistent about:

1. **Synthetic data only.** No file in this repository may contain real
   seller data, real seller-portal exports, or anything derived from a
   proprietary marketplace dataset. Every example and test runs on
   `msmt.data.generate_seller_data` output. If you need a new pattern,
   add it to the synthetic generator rather than committing real data.

2. **No employer references.** Code, docstrings, comments, notebook
   markdown, and app strings should not name specific employers,
   clients, or marketplaces in a way that implies insider knowledge.
   "Walmart Listing Quality Dashboard" or "Amazon Seller Central" as
   public-product references are fine; descriptions of internal
   algorithms or undisclosed weights are not.

3. **Plain English in user-facing strings.** Recommendations, app
   labels, and notebook markdown should read like something an SBDC
   counselor would say to a seller. No data-science jargon without a
   one-sentence definition.

4. **Numpy-style docstrings on public functions.** Every public
   function gets a docstring that explains the *business* meaning, not
   just the parameters. If a parameter has a non-obvious default or a
   subtle constraint, say so.

5. **Practitioner estimates flagged as such.** Where the toolkit uses
   an industry-typical threshold or weight (e.g. the 3.0× suppression
   multiplier, the scorecard signal weights, the classifier's holiday-
   spike ratio), the docstring must explicitly note that the value is
   a practitioner estimate and not a platform-disclosed figure.

6. **Minimal runtime dependencies.** The base library depends only on
   `pandas` and `numpy`. Anything heavier (scipy, matplotlib, prophet,
   streamlit) goes into an optional extra and is imported lazily or
   wrapped in a try-import fallback.

## What kinds of contributions help most

* **Bug reports** with a small reproducible example that runs against
  the synthetic data generator.
* **New worked examples** — additional notebooks that show the
  toolkit applied to a specific seller archetype or marketplace
  scenario.
* **Documentation improvements** — making a docstring clearer for a
  non-technical reader, or expanding the walkthrough notebooks.
* **Threshold tuning** — if you've correlated a signal threshold
  against your own seller's historical performance, a PR that adds
  the alternative as a documented option is welcome.

## What kinds of contributions are out of scope

* Tighter coupling to a specific marketplace's seller-portal API.
* Algorithmic features that would replicate a platform's internal
  ranking or fraud logic. The toolkit's value is being *transparent
  about what it doesn't know*.
* Anything that requires a paid service, a proprietary dataset, or
  credentials.

## License

By contributing, you agree your contributions will be licensed under
the project's MIT license.
