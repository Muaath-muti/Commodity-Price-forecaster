"""
Keeps a Streamlit Community Cloud app awake by actually opening it in a
real (headless) browser, so the JS bundle loads and a genuine WebSocket
session gets established with the Streamlit server.

A plain curl/requests GET does NOT achieve this - it only downloads the
static HTML shell and exits, without ever running the app's Python script.
Streamlit Cloud's sleep-timer only resets on real sessions like this one.

This script also handles the case where the app has ALREADY gone to
sleep: it detects the "Zzzz... this app has gone to sleep" wake screen
and clicks the "Yes, get this app back up!" button, since a sleeping
app can only be woken by that click - not by any kind of passive visit.

NOTE: this deliberately does NOT wait for Playwright's "networkidle"
state. Streamlit keeps a persistent WebSocket connection open the whole
time the page is loaded, so the network is never truly "idle" - waiting
for that condition can hang until it times out even when the page has
loaded fine. Instead, this waits for specific, meaningful things to
appear or disappear (the wake button, or a short grace period).
"""

import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = "https://commodity-price-forecaster.streamlit.app/"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"

# How long to wait for the initial page to finish its basic load.
PAGE_LOAD_TIMEOUT_MS = 90_000
# How long to wait to see if a wake-up button appears at all.
WAKE_BUTTON_CHECK_TIMEOUT_MS = 15_000
# How long to give the app to actually wake up after clicking the button.
WAKE_UP_TIMEOUT_MS = 120_000


def ping_app() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Visiting {APP_URL} ...")
        # "load" just waits for the page's own load event - not for the
        # network to go quiet, which Streamlit's WebSocket never lets happen.
        page.goto(APP_URL, wait_until="load", timeout=PAGE_LOAD_TIMEOUT_MS)

        wake_button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)
        try:
            wake_button.first.wait_for(state="visible", timeout=WAKE_BUTTON_CHECK_TIMEOUT_MS)
            found_wake_button = True
        except PlaywrightTimeoutError:
            found_wake_button = False

        if found_wake_button:
            print("App was asleep - clicking wake-up button...")
            wake_button.first.click()
            try:
                # Wait for the button itself to go away, rather than for
                # the network to go quiet - that's the real signal the
                # app has finished waking up and rendered its next screen.
                wake_button.first.wait_for(state="detached", timeout=WAKE_UP_TIMEOUT_MS)
                print("App has woken up.")
            except PlaywrightTimeoutError:
                print("Clicked wake-up button, but it's still visible after waiting. "
                      "The app may be slow to start - this run will still count as a visit.")
            page.wait_for_timeout(5_000)
        else:
            print("App was already awake (no wake-up button appeared).")

        # A harmless click on the page body to make sure the session
        # registers as a real visit, then hold the session open briefly.
        page.mouse.click(10, 10)
        time.sleep(5)

        browser.close()
        print("Done.")


if __name__ == "__main__":
    try:
        ping_app()
    except Exception as e:
        print(f"Ping failed: {e}", file=sys.stderr)
        sys.exit(1)
