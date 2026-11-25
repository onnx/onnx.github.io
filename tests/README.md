# Browser regression checks

Serve and exercise the actual website with Playwright:

```sh
python -m pip install playwright
python -m playwright install chromium
python -m unittest discover -s tests -v
```

The tests start their own local HTTP server. Network access is needed for the
website's existing Bootstrap and HTML-include scripts. Analytics requests are
blocked. No test dependencies are shipped to the website.

Both an immediately available navigation template and a delayed HTML include
are exercised, since initialization must work in either loading order.

To use an already installed Chrome or Edge, set `BROWSER_CHANNEL` to `chrome` or
`msedge`; the browser-install command is then unnecessary.
