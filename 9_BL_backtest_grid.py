import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# =========================
# Parameters (scalars are fixed across all runs)
# =========================
TRAIN_END_DATE = "2020-12-31"
RIDGE_ALPHA = 10.0  # scalar or list (e.g. [0.1, 1.0, 10.0])

EMBEDDING_PATH = "data/kospidaq_embeddings_OpenAI.xlsx"
RETURN_PATH    = "data/report_return_mapping.xlsx"

RETURN_COLS = [f"log_return_{i}" for i in range(11)]
BEST_RETURN = "log_return_1"  # None for auto-select

# Grid parameters
BASE_UNIVERSE_RETURN_COLS = ["log_return_1", "log_return_5", "log_return_10"]
MIN_REPORT_COUNTS = [10, 20, 30]

# Backtest return key (price-based)
# - If AUTO_MAP_BACKTEST_RETURN is True, the horizon is taken from base_universe_col
#   and mapped to simple_return_k for backtesting.
# - If False, BACKTEST_RETURN_COL is used directly.
AUTO_MAP_BACKTEST_RETURN = True
# - log_return_k: log(px / px.shift(k))
# - simple_return_k: px.pct_change(periods=k)
BACKTEST_RETURN_COL = "simple_return_1"

ADJ_CLOSE_PATH = "data/adj_close_wide_2014_2026.xlsx"
MCAP_PATH      = "data/market_cap_2014_2025.xlsx"

WINDOW = [252, 504]  # scalar or list (e.g. [252, 504])
MIN_COVERAGE = 0.90
RISK_AVERSION = 2.5
TAU = 0.025  # None => 1/n

BACKTEST_START_DATE = "2015-12-31"  # None => after TRAIN_END_DATE
BACKTEST_END_DATE   = None

LONG_ONLY = True
WEIGHT_CLIP = None  # e.g. 0.05

# Transaction costs
TRANSACTION_COST_BPS = 0.0  # fees (both sides)
SELL_TAX_BPS = 0.0          # sell tax only

ANNUALIZATION = 252

# Outputs
OUTPUT_SUMMARY_PATH = "backtest_grid_summary.xlsx"
SAVE_WEIGHTS = False
WEIGHTS_OUTPUT_DIR = "backtest_weights_grid"


# =========================
# Helpers
# =========================

def compute_log_returns(price_wide: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_wide / price_wide.shift(1))


from sklearn.covariance import LedoitWolf

def compute_prior_at_date(
    view_date: pd.Timestamp,
    px_wide: pd.DataFrame,
    mc_wide: pd.DataFrame,
    window: int = 252,
    min_coverage: float = 0.90,
    risk_aversion: float = 2.5,
    ret_wide: pd.DataFrame | None = None,
    universe_filter: list | None = None,
):
    view_date = pd.to_datetime(view_date)

    if view_date not in mc_wide.index:
        raise ValueError("market cap missing on view_date")

    ret = compute_log_returns(px_wide) if ret_wide is None else ret_wide

    if view_date not in ret.index:
        raise ValueError("return missing on view_date")

    end_loc = ret.index.get_loc(view_date)
    start_loc = end_loc - window + 1
    if start_loc < 0:
        raise ValueError("window too long")

    ret_win = ret.iloc[start_loc:end_loc + 1]

    min_obs = int(np.ceil(window * min_coverage))
    valid_obs = ret_win.notna().sum(axis=0)
    tickers_cov = valid_obs[valid_obs >= min_obs].index

    mcap_t = mc_wide.loc[view_date]
    tickers_mcap = mcap_t.dropna().index

    if universe_filter is not None:
        universe_filter = set(universe_filter)
        tickers_cov = [t for t in tickers_cov if t in universe_filter]
        tickers_mcap = [t for t in tickers_mcap if t in universe_filter]

    tickers_univ = sorted(list(set(tickers_cov).intersection(set(tickers_mcap))))
    if len(tickers_univ) < 2:
        raise ValueError("universe too small")

    X = ret_win[tickers_univ].dropna(axis=0, how="any")
    if len(X) < min_obs:
        raise ValueError("not enough common observations")

    lw = LedoitWolf().fit(X.values)
    Sigma = lw.covariance_

    mcap_vec = mc_wide.loc[view_date, tickers_univ].values.astype(float)
    w_mkt = mcap_vec / np.nansum(mcap_vec)

    Pi = risk_aversion * (Sigma @ w_mkt)
    return tickers_univ, Sigma, Pi, w_mkt


def align_view_to_prior(
    df_view: pd.DataFrame,
    view_date: pd.Timestamp,
    tickers_univ: list,
    df_view_by_date: dict | None = None,
):
    view_date = pd.to_datetime(view_date)

    if df_view_by_date is not None:
        tmp = df_view_by_date.get(view_date)
        if tmp is None:
            raise ValueError("view missing on date")
        tmp = tmp.copy()
    else:
        tmp = df_view[df_view["date"] == view_date].copy()

    tmp["ticker_code"] = tmp["ticker_code"].astype(str).str.zfill(6)

    view_tickers_all = set(tmp["ticker_code"].unique())
    prior_tickers = set(tickers_univ)

    tmp = tmp[tmp["ticker_code"].isin(prior_tickers)]

    view_tickers = sorted(tmp["ticker_code"].unique())
    if len(view_tickers) < 1:
        raise ValueError("no view in universe")

    tmp_c = tmp.sort_values("ticker_code")
    Q = tmp_c["pred_return"].values
    tickers_view = tmp_c["ticker_code"].values

    return {
        "tickers_view": tickers_view,
        "Q": Q,
        "missing_in_prior": sorted(list(view_tickers_all - prior_tickers)),
        "missing_in_view": sorted(list(prior_tickers - set(view_tickers))),
        "n_view": len(view_tickers),
        "n_prior": len(prior_tickers),
        "n_common": len(view_tickers),
    }


def compute_oos_mse_for_omega(df_pred, target_return, train_end_date):
    mask = df_pred["date"] > pd.to_datetime(train_end_date)
    y_true = df_pred.loc[mask, target_return].values
    y_pred = df_pred.loc[mask, "pred_return"].values
    return mean_squared_error(y_true, y_pred)


def black_litterman_posterior(Pi, Sigma, P, Q, Omega, tau=0.05):
    Sigma_t = tau * Sigma
    A = np.linalg.inv(Sigma_t) + P.T @ np.linalg.inv(Omega) @ P
    b = np.linalg.inv(Sigma_t) @ Pi + P.T @ np.linalg.inv(Omega) @ Q
    return np.linalg.solve(A, b)


def compute_bl_weights(mu_bl, Sigma, risk_aversion, long_only=True, weight_clip=None):
    w = np.linalg.solve(Sigma, mu_bl) / risk_aversion
    if long_only:
        w = np.clip(w, 0.0, None)
    if weight_clip is not None:
        if long_only:
            w = np.clip(w, 0.0, weight_clip)
        else:
            w = np.clip(w, -weight_clip, weight_clip)
    s = w.sum()
    if np.isclose(s, 0):
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w


def calc_mdd(cum):
    roll_max = np.maximum.accumulate(cum)
    dd = cum / roll_max - 1.0
    return dd.min(), dd


def run_ridge_full_eval(df, embedding_cols, target_return, train_end_date, alpha=10.0):
    df_tmp = df.dropna(subset=[target_return]).copy()
    train_mask = df_tmp["date"] <= train_end_date
    test_mask = df_tmp["date"] > train_end_date

    X_train = df_tmp.loc[train_mask, embedding_cols].values
    y_train = df_tmp.loc[train_mask, target_return].values
    X_test = df_tmp.loc[test_mask, embedding_cols].values
    y_test = df_tmp.loc[test_mask, target_return].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "target": target_return,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_mse": mean_squared_error(y_train, y_pred_train),
        "test_mse": mean_squared_error(y_test, y_pred_test),
    }


def run_ridge_and_predict(df, embedding_cols, target_return, train_end_date, alpha=10.0):
    df_tmp = df.dropna(subset=[target_return]).copy()

    train_mask = df_tmp["date"] <= train_end_date
    test_mask = df_tmp["date"] > train_end_date

    X_train = df_tmp.loc[train_mask, embedding_cols].values
    y_train = df_tmp.loc[train_mask, target_return].values
    X_test = df_tmp.loc[test_mask, embedding_cols].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)

    df_tmp.loc[train_mask, "pred_return"] = model.predict(X_train)
    df_tmp.loc[test_mask, "pred_return"] = model.predict(X_test)

    return df_tmp, model


# =========================
# Main
# =========================

def main():
    # Load data
    df_embed = pd.read_excel(EMBEDDING_PATH)
    df_ret = pd.read_excel(RETURN_PATH)

    embedding_cols = [c for c in df_embed.columns if c.startswith("embedding_")]

    df = pd.concat(
        [
            df_embed[["date", "ticker"] + embedding_cols],
            df_ret[["ticker_code"] + RETURN_COLS],
        ],
        axis=1,
    )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker_code"] = df["ticker_code"].astype(str).str.zfill(6)

    group_cols = ["date", "ticker_code"]
    agg_dict = {c: "mean" for c in embedding_cols}
    for c in RETURN_COLS:
        agg_dict[c] = "mean"

    df_agg = (
        df
        .groupby(group_cols, as_index=False)
        .agg(agg_dict)
    )

    # Load prices and mcap
    px = pd.read_excel(Path(ADJ_CLOSE_PATH), index_col=0)
    mc = pd.read_excel(Path(MCAP_PATH), index_col=0)

    px.index = pd.to_datetime(px.index)
    mc.index = pd.to_datetime(mc.index)

    px.columns = px.columns.astype(str).str.zfill(6)
    mc.columns = mc.columns.astype(str).str.zfill(6)

    px = px.sort_index()
    mc = mc.sort_index()

    # Precompute returns
    ret_log = compute_log_returns(px)

    ret_index = px.index

    summary_rows = []

    if SAVE_WEIGHTS:
        Path(WEIGHTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    ridge_alphas = RIDGE_ALPHA if isinstance(RIDGE_ALPHA, (list, tuple, np.ndarray)) else [RIDGE_ALPHA]
    windows = WINDOW if isinstance(WINDOW, (list, tuple, np.ndarray)) else [WINDOW]

    for alpha in ridge_alphas:
        best_return = BEST_RETURN
        if best_return is None:
            results = []
            for ret in RETURN_COLS:
                out = run_ridge_full_eval(
                    df=df_agg,
                    embedding_cols=embedding_cols,
                    target_return=ret,
                    train_end_date=TRAIN_END_DATE,
                    alpha=alpha,
                )
                results.append(out)
            df_eval = (
                pd.DataFrame(results)
                .sort_values("test_r2", ascending=False)
                .reset_index(drop=True)
            )
            best_return = df_eval.loc[0, "target"]

        df_pred, _ = run_ridge_and_predict(
            df=df_agg,
            embedding_cols=embedding_cols,
            target_return=best_return,
            train_end_date=TRAIN_END_DATE,
            alpha=alpha,
        )

        df_view = (
            df_pred
            .groupby(["date", "ticker_code"], as_index=False)
            .agg({"pred_return": "mean"})
        )

        df_view_by_date = {pd.to_datetime(d): g for d, g in df_view.groupby("date")}

        all_view_dates = pd.DatetimeIndex(df_view["date"].unique()).sort_values()

        if BACKTEST_START_DATE is None:
            start_base = pd.to_datetime(TRAIN_END_DATE)
            start_candidates = all_view_dates[all_view_dates > start_base]
            if len(start_candidates) == 0:
                raise ValueError("BACKTEST_START_DATE not found")
            backtest_start = start_candidates[0]
        else:
            backtest_start = pd.to_datetime(BACKTEST_START_DATE)

        if BACKTEST_END_DATE is None:
            backtest_end = all_view_dates[-1]
        else:
            backtest_end = pd.to_datetime(BACKTEST_END_DATE)

        view_dates = [d for d in all_view_dates if (d >= backtest_start) and (d <= backtest_end)]

        oos_mse = compute_oos_mse_for_omega(
            df_pred=df_pred,
            target_return=best_return,
            train_end_date=TRAIN_END_DATE,
        )

        for col in BASE_UNIVERSE_RETURN_COLS:
            df_base = df_agg.dropna(subset=[col]).copy()
            df_base["ticker_code"] = df_base["ticker_code"].astype(str).str.zfill(6)
            report_count = df_base["ticker_code"].value_counts()

            for min_cnt in MIN_REPORT_COUNTS:
                base_universe = sorted(report_count[report_count >= min_cnt].index.tolist())

                for window in windows:
                    if AUTO_MAP_BACKTEST_RETURN:
                        ret_col = f"simple_return_{int(col.split('_')[-1])}"
                    else:
                        ret_col = BACKTEST_RETURN_COL

                    if ret_col.startswith("log_return_"):
                        horizon = int(ret_col.split("_")[-1])
                        if horizon < 1:
                            raise ValueError("BACKTEST_RETURN_COL horizon must be >= 1")
                        ret_wide = np.log(px / px.shift(horizon))
                    elif ret_col.startswith("simple_return_"):
                        horizon = int(ret_col.split("_")[-1])
                        if horizon < 1:
                            raise ValueError("BACKTEST_RETURN_COL horizon must be >= 1")
                        ret_wide = px.pct_change(periods=horizon)
                    else:
                        raise ValueError("BACKTEST_RETURN_COL must be log_return_k or simple_return_k")
                    ret_index = ret_wide.index

                    t0 = time.time()

                    results = []
                    weights_hist = []

                    prev_w = None
                    prev_tickers = None

                    for view_date in tqdm(view_dates, desc=f"BT a={alpha} w={window} {col}/{min_cnt}"):
                        try:
                            tickers_univ, Sigma, Pi, w_mkt = compute_prior_at_date(
                                view_date=view_date,
                                px_wide=px,
                                mc_wide=mc,
                                window=window,
                                min_coverage=MIN_COVERAGE,
                                risk_aversion=RISK_AVERSION,
                                ret_wide=ret_log,
                                universe_filter=base_universe,
                            )

                            aligned = align_view_to_prior(
                                df_view=df_view,
                                view_date=view_date,
                                tickers_univ=tickers_univ,
                                df_view_by_date=df_view_by_date,
                            )

                            Q = aligned["Q"]
                            tickers_view = aligned["tickers_view"]

                            n = len(tickers_univ)
                            k = len(Q)
                            idx_map = {t: i for i, t in enumerate(tickers_univ)}

                            P = np.zeros((k, n))
                            for i, t in enumerate(tickers_view):
                                P[i, idx_map[t]] = 1.0

                            tau = (1.0 / n) if TAU is None else TAU
                            Omega = np.eye(k) * oos_mse

                            mu_bl = black_litterman_posterior(
                                Pi=Pi,
                                Sigma=Sigma,
                                P=P,
                                Q=Q,
                                Omega=Omega,
                                tau=tau,
                            )

                            w = compute_bl_weights(
                                mu_bl=mu_bl,
                                Sigma=Sigma,
                                risk_aversion=RISK_AVERSION,
                                long_only=LONG_ONLY,
                                weight_clip=WEIGHT_CLIP,
                            )

                            bm_sum = np.sum(w_mkt)
                            if np.isclose(bm_sum, 0):
                                w_bm = np.ones_like(w_mkt) / len(w_mkt)
                            else:
                                w_bm = w_mkt / bm_sum

                        except Exception:
                            continue

                        if view_date not in ret_index:
                            continue

                        loc = ret_index.get_loc(view_date)
                        if not isinstance(loc, (int, np.integer)):
                            loc = loc.start

                        ret_pos = loc + horizon
                        if ret_pos >= len(ret_index):
                            continue

                        ret_date = ret_index[ret_pos]
                        ret_vec = ret_wide.loc[ret_date, tickers_univ].values.astype(float)
                        tickers_ret = np.array(tickers_univ)

                        if np.isnan(ret_vec).any():
                            mask = ~np.isnan(ret_vec)
                            if mask.sum() < 2:
                                continue
                            ret_vec = ret_vec[mask]
                            w = w[mask]
                            w = w / w.sum()
                            w_bm = w_bm[mask]
                            w_bm = w_bm / w_bm.sum()
                            tickers_ret = tickers_ret[mask]

                        turnover = 0.0
                        sell_turnover = 0.0
                        if prev_w is None:
                            turnover = np.sum(np.abs(w))
                            sell_turnover = np.sum(w)
                        else:
                            if (prev_tickers is not None) and np.array_equal(prev_tickers, tickers_ret):
                                delta_w = w - prev_w
                                turnover = np.sum(np.abs(delta_w))
                                sell_turnover = np.sum(np.clip(-delta_w, 0.0, None))
                            else:
                                turnover = np.sum(np.abs(w))
                                sell_turnover = np.sum(w)

                        cost = turnover * (TRANSACTION_COST_BPS / 10000.0) + sell_turnover * (SELL_TAX_BPS / 10000.0)

                        port_ret = float(np.dot(w, ret_vec)) - cost
                        bm_ret = float(np.dot(w_bm, ret_vec))

                        results.append({
                            "view_date": view_date,
                            "next_date": ret_date,
                            "net_ret": port_ret,
                            "bm_ret": bm_ret,
                            "cost": cost,
                            "n_assets": len(w),
                        })

                        if SAVE_WEIGHTS:
                            weights_hist.append(
                                pd.DataFrame({
                                    "date": view_date,
                                    "ticker_code": tickers_ret,
                                    "weight": w,
                                })
                            )

                        prev_w = w
                        prev_tickers = tickers_ret

                    if len(results) == 0:
                        df_bt = pd.DataFrame()
                    else:
                        df_bt = pd.DataFrame(results).sort_values("view_date").set_index("view_date")

                    if df_bt.empty:
                        summary_rows.append({
                            "ridge_alpha": alpha,
                            "window": window,
                            "base_universe_col": col,
                            "min_report_count": min_cnt,
                            "n_base_universe": len(base_universe),
                            "n_bt_days": 0,
                            "sharpe": np.nan,
                            "mdd": np.nan,
                            "cum_return": np.nan,
                            "mean_daily": np.nan,
                            "daily_vol": np.nan,
                            "elapsed_sec": time.time() - t0,
                        })
                        continue

                    cum = (1.0 + df_bt["net_ret"]).cumprod()
                    mdd, _ = calc_mdd(cum.values)

                    ret_std = df_bt["net_ret"].std()
                    if ret_std == 0 or np.isnan(ret_std):
                        sharpe = np.nan
                    else:
                        sharpe = (df_bt["net_ret"].mean() / ret_std) * np.sqrt(ANNUALIZATION)

                    summary_rows.append({
                        "ridge_alpha": alpha,
                        "window": window,
                        "base_universe_col": col,
                        "min_report_count": min_cnt,
                        "n_base_universe": len(base_universe),
                        "n_bt_days": len(df_bt),
                        "sharpe": sharpe,
                        "mdd": mdd,
                        "cum_return": cum.iloc[-1] - 1.0,
                        "mean_daily": df_bt["net_ret"].mean(),
                        "daily_vol": ret_std,
                        "elapsed_sec": time.time() - t0,
                    })

                    if SAVE_WEIGHTS and len(weights_hist) > 0:
                        weights_history = pd.concat(weights_hist, ignore_index=True)
                        weights_wide = (
                            weights_history
                            .pivot_table(index="date", columns="ticker_code", values="weight", aggfunc="mean")
                            .reindex(columns=base_universe, fill_value=0.0)
                            .sort_index()
                        )
                        weights_wide["portfolio_return"] = df_bt["net_ret"].reindex(weights_wide.index)

                        out_path = Path(WEIGHTS_OUTPUT_DIR) / f"weights_{col}_{min_cnt}_a{alpha}_w{window}.xlsx"
                        weights_wide.to_excel(out_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(OUTPUT_SUMMARY_PATH, index=False)
    print("Saved:", OUTPUT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
