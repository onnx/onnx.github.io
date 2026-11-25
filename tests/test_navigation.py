# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import http.server
import os
import re
import threading
import time
import unittest
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/index.html":
            # Exercise the fast-header case deterministically, using the real
            # template. ?late-header exercises the actual asynchronous include.
            root = Path(self.directory)
            body = (root / "index.html").read_text(encoding="utf-8").replace(
                '<div w3-include-html="partials/navigation.html"></div>',
                (root / "partials/navigation.html").read_text(encoding="utf-8"),
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/partials/navigation.html":
            time.sleep(0.5)
        super().do_GET()

    def log_message(self, *_args):
        pass

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Navigating away can cancel an in-flight asset response.
            pass


class NavigationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root))
        )
        cls.addClassCleanup(cls.server.server_close)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.server.shutdown)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch(
            channel=os.environ.get("BROWSER_CHANNEL") or None
        )
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 390, "height": 844})
        self.addCleanup(self.context.close)
        # Do not send analytics events while exercising the local website.
        self.context.route("**/googletagmanager.com/**", lambda route: route.abort())
        self.page = self.context.new_page()
        self.page.set_default_timeout(10000)
        self.page.goto(self.url + "/index.html", wait_until="load")
        expect(self.page.locator("#navigation")).to_be_attached()
        self.page.wait_for_function("typeof bootstrap !== 'undefined'")

    def open_menu(self):
        self.page.get_by_role("button", name="Toggle navigation").click()
        expect(self.page.locator("#navbarNav")).to_have_class(
            re.compile(r"\bshow\b")
        )

    def test_announcement_navigates_to_news_article(self):
        for width in (390, 1440):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 900})
                self.page.goto(self.url + "/index.html", wait_until="load")
                link = self.page.locator("#newsSlider a")
                expect(link).to_be_visible()
                destination = link.evaluate("element => element.href")
                link.click()
                expect(self.page).to_have_url(destination)
                article_id = destination.split("#", 1)[1]
                expect(self.page.locator(f"[id='{article_id}']")).to_be_visible()

    def test_mobile_dropdowns_stay_open(self):
        self.open_menu()
        for name in ("Docs", "Community"):
            with self.subTest(menu=name):
                toggle = self.page.get_by_role("button", name=name, exact=True)
                toggle.click()
                expect(toggle).to_have_attribute("aria-expanded", "true")
                # Check after the collapse transition, not just before it starts.
                self.page.wait_for_timeout(500)
                expect(self.page.locator("#navbarNav")).to_be_visible()
                expect(toggle.locator("+ .dropdown-menu")).to_be_visible()
        self.page.get_by_role("link", name="Get Involved", exact=True).first.click()
        expect(self.page).to_have_url(self.url + "/get-involved.html")

    def test_keyboard_can_open_dropdown_and_follow_link(self):
        self.open_menu()
        self.page.get_by_role("button", name="Docs", exact=True).focus()
        self.page.keyboard.press("Enter")
        link = self.page.locator("#navigation").get_by_role("link", name="Supported Tools")
        self.page.wait_for_timeout(500)
        expect(link).to_be_visible()
        link.focus()
        self.page.keyboard.press("Enter")
        expect(self.page).to_have_url(self.url + "/supported-tools.html")

    def test_escape_and_outside_click_close_menu(self):
        self.open_menu()
        self.page.keyboard.press("Escape")
        expect(self.page.locator("#navbarNav")).not_to_be_visible()
        self.open_menu()
        self.page.locator(".shadow-box .shadow").click()
        expect(self.page.locator("#navbarNav")).not_to_be_visible()

    def test_navigation_works_when_header_loads_after_initialization(self):
        errors = []
        self.page.on("pageerror", lambda error: errors.append(str(error)))
        self.page.goto(self.url + "/index.html?late-header", wait_until="load")
        expect(self.page.locator("#navigation")).to_be_attached()
        self.open_menu()
        expect(self.page.locator("header")).to_have_class(re.compile(r"\bheader-collapse\b"))
        self.page.keyboard.press("Escape")
        expect(self.page.locator("#navbarNav")).not_to_be_visible()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
