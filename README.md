# northstar

![tests](https://github.com/OWNER/REPO/actions/workflows/tests.yml/badge.svg)

## Command line utilities

Run ad-hoc calculations or integration checks directly from the project root after installing dependencies inside `app/`:

```
poetry run python -m src.cli calc --symbol KXS.TO
poetry run calc --symbol AAPL --send-telegram
poetry run tgtest --text "Ping"
```
