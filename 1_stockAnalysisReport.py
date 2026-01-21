#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
from urllib.parse import urljoin
from tqdm import tqdm


def _ensure_deps():
    try:
        from selenium import webdriver  # noqa: F401
        from selenium.webdriver.chrome.options import Options  # noqa: F401
        from selenium.webdriver.common.by import By  # noqa: F401
        from selenium.webdriver.support.ui import WebDriverWait  # noqa: F401
        from selenium.webdriver.support import expected_conditions as EC  # noqa: F401
        from selenium.webdriver.chrome.service import Service  # noqa: F401
        from webdriver_manager.chrome import ChromeDriverManager  # noqa: F401
    except Exception:
        print("pip install selenium webdriver-manager pandas lxml", file=sys.stderr)
        raise


_ensure_deps()
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException


BASE = "https://finance.naver.com"
LIST_URL_TPL = (
    "https://finance.naver.com/research/company_list.naver"
    "?keyword=&brokerCode=&searchType=writeDate&writeFromDate={start}"
    "&writeToDate={end}&itemName=&itemCode=&page={page}"
)
DEFAULT_START = "2025-12-22"
DEFAULT_END = "2025-12-23"
DEFAULT_PAGES = 2
DEFAULT_OUTPUT = "data"
DEFAULT_HEADLESS = True


@dataclass
class ReportRow:
    box_type_m: str


def build_driver(headless: bool = True, user_data_dir: str | None = None, remote_debug_port: int | None = None) -> webdriver.Chrome:
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument(f"--user-agent={ua}")
    opts.add_argument("--window-size=1280,1800")
    opts.add_argument("--disable-dev-shm-usage")
    if user_data_dir:
        opts.add_argument(f"--user-data-dir={user_data_dir}")
        opts.add_argument("--no-first-run")
        opts.add_argument("--no-default-browser-check")
    if remote_debug_port:
        opts.add_argument(f"--remote-debugging-port={remote_debug_port}")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def get_report_links(driver: webdriver.Chrome, wait: WebDriverWait, list_url: str) -> List[str]:
    driver.get(list_url)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.type_1")))
    links: List[str] = []
    for a in driver.find_elements(By.CSS_SELECTOR, "a"):
        href = a.get_attribute("href") or ""
        if "company_read.naver" in href:
            links.append(urljoin(BASE, href))
    seen, uniq = set(), []
    for u in links:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def extract_box_type_m(driver: webdriver.Chrome) -> str:
    try:
        el = driver.find_element(By.CSS_SELECTOR, ".box_type_m")
        return el.text.strip()
    except Exception:
        return ""


def parse_report(driver: webdriver.Chrome, wait: WebDriverWait, url: str) -> ReportRow:
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(0.3)
    return ReportRow(box_type_m=extract_box_type_m(driver))


def _split_col(df_in: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if col not in df_in.columns:
        return df_in
    parts = df_in[col].fillna("").astype(str).apply(
        lambda s: [p.strip() for p in s.split("|") if p.strip()]
    )
    max_parts = int(parts.apply(len).max() or 0)
    for i in range(max_parts):
        df_in[f"{prefix}_p{i+1}"] = parts.apply(lambda arr, i=i: arr[i] if i < len(arr) else "")
    return df_in


def crawl(start: str, end: str, pages: int, output_dir: str, headless: bool = True) -> str:
    RESTART_EVERY = 20
    RETRIES = 2

    def build_and_wait():
        d = build_driver(headless=headless)
        w = WebDriverWait(d, 30)
        return d, w

    driver, wait = build_and_wait()
    rows: List[ReportRow] = []
    try:
        for page in tqdm(range(1, pages + 1), desc="List pages", unit="page"):
            # periodic restart for stability
            if page != 1 and (page - 1) % RESTART_EVERY == 0:
                try:
                    driver.quit()
                except Exception:
                    pass
                driver, wait = build_and_wait()

            list_url = LIST_URL_TPL.format(start=start, end=end, page=page)
            # get links with retries
            links: List[str] = []
            for attempt in range(RETRIES + 1):
                try:
                    links = get_report_links(driver, wait, list_url)
                    break
                except (InvalidSessionIdException, WebDriverException):
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver, wait = build_and_wait()
                    if attempt == RETRIES:
                        links = []
                except Exception:
                    links = []
                    break
            for url in tqdm(links, desc=f"Reports p{page}", unit="report", leave=False):
                # per-report retries
                ok = False
                for attempt in range(RETRIES + 1):
                    try:
                        rows.append(parse_report(driver, wait, url))
                        ok = True
                        break
                    except (InvalidSessionIdException, WebDriverException):
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        driver, wait = build_and_wait()
                        if attempt == RETRIES:
                            break
                    except Exception:
                        break
                if not ok:
                    rows.append(ReportRow(box_type_m=""))
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    os.makedirs(output_dir, exist_ok=True)

    # Build per-report lines and keep only line_2..line_6
    per_report: List[List[str]] = []
    for r in rows:
        if not r.box_type_m:
            continue
        split = [ln.strip() for ln in r.box_type_m.splitlines() if ln.strip()]
        if split:
            per_report.append(split)

    desired_cols = [f"line_{i}" for i in range(2, 7)]
    if per_report:
        norm_rows: List[Dict[str, str]] = []
        for lst in per_report:
            row: Dict[str, str] = {}
            for i, col in zip(range(2, 7), desired_cols):
                row[col] = lst[i - 1] if len(lst) >= i else ""
            norm_rows.append(row)
        df = pd.DataFrame(norm_rows, columns=desired_cols)
    else:
        df = pd.DataFrame(columns=desired_cols)

    # Split line_3 and line_4 into parts
    df = _split_col(df, "line_3", "line_3")
    df = _split_col(df, "line_4", "line_4")

    # Select and ensure columns
    final_order = [
        "line_2",
        "line_3_p1",
        "line_3_p2",
        "line_3_p3",
        "line_4_p1",
        "line_4_p2",
        "line_5",
        "line_6",
    ]
    for col in final_order:
        if col not in df.columns:
            df[col] = ""
    df = df[final_order].copy()

    # Temporary English keys
    df = df.rename(
        columns={
            "line_2": "equity",
            "line_3_p1": "broker",
            "line_3_p2": "date",
            "line_3_p3": "views",
            "line_4_p1": "target",
            "line_4_p2": "rating",
            "line_5": "title",
            "line_6": "body",
        }
    )

    # Normalize values (label-agnostic)
    if "views" in df.columns:
        s = df["views"].fillna("").astype(str)
        df["views"] = s.str.extract(r"(\d[\d,]*)", expand=False).fillna(s.str.strip())
    if "target" in df.columns:
        s = df["target"].fillna("").astype(str)
        df["target"] = s.str.extract(r"(\d[\d,]*)", expand=False).fillna(s.str.strip())
    if "rating" in df.columns:
        s = df["rating"].fillna("").astype(str)
        df["rating"] = s.str.replace(r"^\s*[^\s|:：]+\s*[:：]?\s*", "", regex=True).str.strip()

    # Option A: split equity and reconstruct title/content
    parts = df["equity"].fillna("").astype(str).str.split(r"\s+", n=1, regex=True)
    left = parts.str[0].fillna("").str.strip()
    right = parts.str[1].fillna("").str.strip()
    df["equity"] = left
    df["title"] = right.where(right.ne(""), df["title"].fillna(""))
    df["body"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()

    # Final select and rename to English (drop views, title)
    out = df[["equity", "broker", "date", "target", "rating", "body"]].copy()
    out.rename(
        columns={
            "equity": "ticker",
            "broker": "broker",
            "date": "date",
            "target": "target_price",
            "rating": "rating",
            "body": "content",
        },
        inplace=True,
    )

    parquet_path = os.path.join(output_dir, f"stockAnalysisReport_{start}_{end}.parquet")
    out.to_parquet(parquet_path, index=False)
    return parquet_path


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Naver Finance research crawler (company)")
    p.add_argument("--start", default=DEFAULT_START, help="start date YYYY-MM-DD")
    p.add_argument("--end", default=DEFAULT_END, help="end date YYYY-MM-DD")
    p.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="number of list pages to crawl")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="directory to save CSV")
    p.add_argument("--no-headless", action="store_true", help="run Chrome with UI")
    args = p.parse_args(argv)

    try:
        headless = DEFAULT_HEADLESS if not args.no_headless else False
        path = crawl(
            start=args.start,
            end=args.end,
            pages=max(1, args.pages),
            output_dir=args.output,
            headless=headless,
        )
        print(f"Saved: {path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
