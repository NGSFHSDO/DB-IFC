from __future__ import annotations

import time
from http.cookiejar import MozillaCookieJar
from pathlib import Path

import pandas as pd
import requests
from pykrx import bond, stock


START_DATE = "20130102"
END_DATE = "20251230"
KOSPI_INDEX_TICKER = "1001"

OUTPUT_DIR = Path("data")
OUTPUT_XLSX = OUTPUT_DIR / f"kospi_krx_{START_DATE}_{END_DATE}.xlsx"
OUTPUT_CSV = OUTPUT_DIR / f"kospi_krx_{START_DATE}_{END_DATE}.csv"
OUTPUT_TREASURY_XLSX = OUTPUT_DIR / f"treasury3m_proxy_krx_{START_DATE}_{END_DATE}.xlsx"
OUTPUT_TREASURY_CSV = OUTPUT_DIR / f"treasury3m_proxy_krx_{START_DATE}_{END_DATE}.csv"
COOKIE_CANDIDATES = [Path("krx_cookies.txt"), Path("cookies.txt")]

# KRX 장외채권 수익률 코드
# Use KRX OTC treasury yield series: 국고채 1년 (code=3006)
RISK_FREE_KIND_CODE = "3006"
RISK_FREE_KIND_NAME_PYKRX = "국고채1년"
RISK_FREE_KIND_DISPLAY = "국고채 1년"

# Optional manual override for KRX session.
JSESSIONID_OVERRIDE = (
    "Uq7wqePMRp2tm23V9dxy715WsBl3CzjVeseKnbYbirkHnW1ttpQrZBjfp5h129DL."
    "bWRjX2RvbWFpbi9tZGNvd2FwMS1tZGNhcHAwMQ=="
)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # Case 1: low-level KRX raw output keys
    raw_cols = {
        "TRD_DD",
        "OPNPRC_IDX",
        "HGPRC_IDX",
        "LWPRC_IDX",
        "CLSPRC_IDX",
        "ACC_TRDVOL",
        "ACC_TRDVAL",
    }
    if raw_cols.issubset(set(out.columns)):
        out["date"] = pd.to_datetime(out["TRD_DD"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        rename_map = {
            "OPNPRC_IDX": "open",
            "HGPRC_IDX": "high",
            "LWPRC_IDX": "low",
            "CLSPRC_IDX": "close",
            "ACC_TRDVOL": "volume",
            "ACC_TRDVAL": "trading_value",
            "MKTCAP": "market_cap",
        }
        keep_cols = [c for c in rename_map if c in out.columns]
        out = out[keep_cols].rename(columns=rename_map)
        for c in out.columns:
            out[c] = _to_numeric_series(out[c])
        out.index.name = "date"
        return out

    # Case 2: stock.get_index_ohlcv_by_date output (possibly Korean column names)
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    ordered_names: list[str] = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "market_cap",
    ]
    rename_by_pos = {
        src: dst for src, dst in zip(list(out.columns), ordered_names, strict=False)
    }
    out = out.rename(columns=rename_by_pos)
    for c in out.columns:
        out[c] = _to_numeric_series(out[c])
    out.index.name = "date"
    return out


def _normalize_treasury_1y(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # Low-level KRX output
    if {"DISCLS_DD", "LST_ORD_BAS_YD"}.issubset(set(out.columns)):
        out["date"] = pd.to_datetime(out["DISCLS_DD"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        rename_map = {
            "LST_ORD_BAS_YD": "yield_3m_proxy",
            "CMP_YD": "delta",
        }
        keep_cols = [c for c in rename_map if c in out.columns]
        out = out[keep_cols].rename(columns=rename_map)
        for c in out.columns:
            out[c] = _to_numeric_series(out[c])
        out.index.name = "date"
        return out

    # pykrx.bond wrapped output
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if len(out.columns) >= 1:
        rename = {out.columns[0]: "yield_3m_proxy"}
        if len(out.columns) >= 2:
            rename[out.columns[1]] = "delta"
        out = out.rename(columns=rename)
    keep = [c for c in ["yield_1y", "delta"] if c in out.columns]
    out = out[keep]
    for c in out.columns:
        out[c] = _to_numeric_series(out[c])
    out.index.name = "date"
    return out


def _fetch_low_level_krx(start_date: str, end_date: str, index_ticker: str) -> pd.DataFrame:
    # Fallback when stock.get_index_ohlcv_by_date fails due package/index-name issues.
    from pykrx.website.krx.market import core

    target_name = "\\uac1c\\ubcc4\\uc9c0\\uc218\\uc2dc\\uc138"  # 개별지수시세
    cls = None
    for name, obj in core.__dict__.items():
        if isinstance(obj, type) and name.encode("unicode_escape").decode() == target_name:
            cls = obj
            break

    if cls is None:
        raise RuntimeError("Failed to locate low-level KRX class for index OHLCV.")

    # pykrx internally maps '1001' -> indIdx='1', indIdx2='001'
    group_id = index_ticker[0]
    ticker = index_ticker[1:]
    raw = cls().fetch(ticker=ticker, group_id=group_id, fromdate=start_date, todate=end_date)
    return raw


def _fetch_low_level_treasury_1y(start_date: str, end_date: str) -> pd.DataFrame:
    # bndKindTpCd=3006 -> 국고채 1년
    from pykrx.website.krx.bond import core

    target_name = "\\uac1c\\ubcc4\\ucd94\\uc774_\\uc7a5\\uc678\\ucc44\\uad8c\\uc218\\uc775\\ub960"  # 개별추이_장외채권수익률
    cls = None
    for name, obj in core.__dict__.items():
        if isinstance(obj, type) and name.encode("unicode_escape").decode() == target_name:
            cls = obj
            break

    if cls is None:
        raise RuntimeError("Failed to locate low-level KRX class for treasury yields.")

    raw = cls().fetch(strtDd=start_date, endDd=end_date, bndKindTpCd=RISK_FREE_KIND_CODE)
    return raw


def _load_krx_cookies(cookies_path: Path) -> requests.cookies.RequestsCookieJar:
    jar = requests.cookies.RequestsCookieJar()
    if not cookies_path.exists():
        if JSESSIONID_OVERRIDE:
            jar.set("JSESSIONID", JSESSIONID_OVERRIDE, domain=".krx.co.kr", path="/")
            jar.set("JSESSIONID", JSESSIONID_OVERRIDE, domain="data.krx.co.kr", path="/")
        return jar

    moz = MozillaCookieJar(str(cookies_path))
    moz.load(ignore_discard=True, ignore_expires=True)

    # Prefer manual override -> data.krx.co.kr -> .krx.co.kr -> www.krx.co.kr
    preferred_jsession = JSESSIONID_OVERRIDE
    jsession_by_domain = {}
    for c in moz:
        if c.name == "JSESSIONID" and "krx.co.kr" in c.domain:
            jsession_by_domain[c.domain] = c.value
    if not preferred_jsession:
        preferred_jsession = (
            jsession_by_domain.get("data.krx.co.kr")
            or jsession_by_domain.get(".krx.co.kr")
            or jsession_by_domain.get("www.krx.co.kr")
        )

    for c in moz:
        if "krx.co.kr" not in c.domain:
            continue
        if c.name == "JSESSIONID":
            if not preferred_jsession:
                continue
            # Normalize to a single session value to avoid conflicts.
            if c.value != preferred_jsession:
                continue
        jar.set(
            c.name,
            c.value,
            domain=c.domain,
            path=c.path,
        )

    if preferred_jsession:
        # Ensure KRX data endpoint receives JSESSIONID regardless of source domain.
        jar.set("JSESSIONID", preferred_jsession, domain=".krx.co.kr", path="/")
        jar.set("JSESSIONID", preferred_jsession, domain="data.krx.co.kr", path="/")
    return jar


def _iter_date_chunks(start_date: str, end_date: str, max_days: int = 730):
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    step = pd.Timedelta(days=max_days)
    one_day = pd.Timedelta(days=1)
    while s <= e:
        t = min(s + step, e)
        yield s.strftime("%Y%m%d"), t.strftime("%Y%m%d")
        s = t + one_day


def _fetch_krx_with_cookies(
    start_date: str,
    end_date: str,
    index_ticker: str,
    cookies_path: Path,
) -> pd.DataFrame:
    session = requests.Session()
    session.cookies.update(_load_krx_cookies(cookies_path))

    # Keep both domain-specific and root referers.
    warmup_urls = [
        "https://data.krx.co.kr/",
        "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010101",
    ]
    for u in warmup_urls:
        try:
            session.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        except Exception:
            pass

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010101",
        "Origin": "https://data.krx.co.kr",
        "X-Requested-With": "XMLHttpRequest",
    }

    group_id = index_ticker[0]
    ticker = index_ticker[1:]
    endpoint = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    output_rows = []
    for s, e in _iter_date_chunks(start_date, end_date, max_days=730):
        payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT00301",
            "indIdx2": ticker,
            "indIdx": group_id,
            "strtDd": s,
            "endDd": e,
        }
        resp = session.post(endpoint, headers=headers, data=payload, timeout=30)
        text = (resp.text or "").strip()
        if resp.status_code != 200 or text == "LOGOUT":
            raise RuntimeError(
                f"KRX cookie request failed ({resp.status_code}) for {s}~{e}: {text[:80]}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"KRX returned non-JSON response for {s}~{e}: {text[:120]}"
            ) from exc

        rows = data.get("output", [])
        if isinstance(rows, list):
            output_rows.extend(rows)

        time.sleep(0.2)

    return pd.DataFrame(output_rows)


def _fetch_treasury_1y_with_cookies(
    start_date: str,
    end_date: str,
    cookies_path: Path,
) -> pd.DataFrame:
    session = requests.Session()
    session.cookies.update(_load_krx_cookies(cookies_path))

    warmup_urls = [
        "https://data.krx.co.kr/",
        "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201040101",
    ]
    for u in warmup_urls:
        try:
            session.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        except Exception:
            pass

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201040101",
        "Origin": "https://data.krx.co.kr",
        "X-Requested-With": "XMLHttpRequest",
    }
    endpoint = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    output_rows = []
    for s, e in _iter_date_chunks(start_date, end_date, max_days=365):
        payload = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT11402",
            "inqTpCd": "E",
            "strtDd": s,
            "endDd": e,
            "bndKindTpCd": RISK_FREE_KIND_CODE,
        }
        resp = session.post(endpoint, headers=headers, data=payload, timeout=30)
        text = (resp.text or "").strip()
        if resp.status_code != 200 or text in {"LOGOUT", "INVALIDPERIOD2"}:
            raise RuntimeError(
                f"KRX treasury request failed ({resp.status_code}) for {s}~{e}: {text[:80]}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"KRX treasury response is not JSON for {s}~{e}: {text[:120]}"
            ) from exc

        rows = data.get("output", [])
        if isinstance(rows, list):
            output_rows.extend(rows)

        time.sleep(0.2)

    return pd.DataFrame(output_rows)


def fetch_kospi_krx(start_date: str, end_date: str, index_ticker: str = KOSPI_INDEX_TICKER) -> pd.DataFrame:
    # Primary path
    try:
        df = stock.get_index_ohlcv_by_date(
            fromdate=start_date,
            todate=end_date,
            ticker=index_ticker,
            name_display=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        # Fallback path 1: cookie-backed direct KRX request
        try:
            df = _fetch_krx_with_cookies(
                start_date=start_date,
                end_date=end_date,
                index_ticker=index_ticker,
                cookies_path=next((p for p in COOKIE_CANDIDATES if p.exists()), COOKIE_CANDIDATES[0]),
            )
        except Exception as exc:
            # Fallback path 2: pykrx low-level class
            try:
                df = _fetch_low_level_krx(start_date, end_date, index_ticker)
            except Exception as low_exc:
                raise RuntimeError(
                    "KRX request failed. Even cookie-backed request did not pass. "
                    "Check whether JSESSIONID in cookies.txt is fresh for data.krx.co.kr."
                ) from low_exc

    out = _normalize_ohlcv(df)
    if out.empty:
        raise RuntimeError(
            "KRX returned an empty dataset. Verify connectivity/IP access to KRX and retry."
        )
    return out


def fetch_treasury_1y_krx(start_date: str, end_date: str) -> pd.DataFrame:
    # Primary path
    try:
        df = bond.get_otc_treasury_yields(start_date, end_date, RISK_FREE_KIND_NAME_PYKRX)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        # Fallback path 1: cookie-backed direct KRX request
        try:
            df = _fetch_treasury_1y_with_cookies(
                start_date=start_date,
                end_date=end_date,
                cookies_path=next((p for p in COOKIE_CANDIDATES if p.exists()), COOKIE_CANDIDATES[0]),
            )
        except Exception:
            # Fallback path 2: pykrx low-level class
            try:
                df = _fetch_low_level_treasury_1y(start_date, end_date)
            except Exception as low_exc:
                raise RuntimeError(
                    "KRX treasury request failed. Check JSESSIONID/cookie freshness."
                ) from low_exc

    out = _normalize_treasury_1y(df)
    if out.empty:
        raise RuntimeError("KRX treasury endpoint returned empty data.")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cookie_path = next((p for p in COOKIE_CANDIDATES if p.exists()), COOKIE_CANDIDATES[0])
    ck = _load_krx_cookies(cookie_path)
    jsession = [c for c in ck if c.name == "JSESSIONID"]
    print(f"cookie file: {cookie_path}")
    print(f"KRX cookies loaded: {len(ck)} / JSESSIONID count: {len(jsession)}")

    df = fetch_kospi_krx(START_DATE, END_DATE, KOSPI_INDEX_TICKER)
    rf = fetch_treasury_1y_krx(START_DATE, END_DATE)

    df.to_excel(OUTPUT_XLSX)
    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    rf.to_excel(OUTPUT_TREASURY_XLSX)
    rf.to_csv(OUTPUT_TREASURY_CSV, encoding="utf-8-sig")

    print("Fetched KOSPI index data from KRX")
    print(f"rows: {len(df)}")
    print(f"date range: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"saved xlsx: {OUTPUT_XLSX}")
    print(f"saved csv : {OUTPUT_CSV}")
    print(df.head(3))
    print(df.tail(3))
    print("")
    print(f"Fetched KRX short-rate proxy: {RISK_FREE_KIND_DISPLAY}")
    print(f"rows: {len(rf)}")
    print(f"date range: {rf.index.min().date()} ~ {rf.index.max().date()}")
    print(f"saved xlsx: {OUTPUT_TREASURY_XLSX}")
    print(f"saved csv : {OUTPUT_TREASURY_CSV}")
    print(rf.head(3))
    print(rf.tail(3))


if __name__ == "__main__":
    main()
