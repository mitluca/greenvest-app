import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenVest | ESG Portfolio Optimiser",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 600; }
    div[data-testid="metric-container"] {
        background: #f7f4ef;
        border: 1px solid #d4e8d4;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Pure functions (unchanged from original) ───────────────────────────────────

def p_return(w1, mu1, mu2):
    return w1 * mu1 + (1 - w1) * mu2

def p_std(w1, s1, s2, rho):
    return np.sqrt(w1**2*s1**2 + (1-w1)**2*s2**2 + 2*rho*w1*(1-w1)*s1*s2)

def p_esg(w1, e1, e2):
    return w1 * e1 + (1 - w1) * e2

def p_util(mu, sig, esg, g, l):
    return mu - (g / 2) * sig**2 + l * esg

def composite_esg(e, s, g, we, ws, wg):
    t = we + ws + wg
    return (we * e + ws * s + wg * g) / t if t > 0 else 0

def esg_momentum(now, last):
    return (now - last) / last if last > 0 else 0

def esg_rating(score):
    for thresh, rating in [(0.85,"AAA"),(0.70,"AA"),(0.55,"A"),
                           (0.40,"BBB"),(0.25,"BB"),(0.10,"B")]:
        if score >= thresh:
            return rating
    return "CCC"

def esg_sharpe(mu, rf, sig, lam, esg):
    return (mu - rf + lam * esg) / sig if sig > 0 else float("nan")

def future_value(pv, r, years):
    return pv * (1 + r) ** years

def optimise(mu1, mu2, s1, s2, e1, e2, rho, gamma, lam, n=2000):
    w = np.linspace(0, 1, n)
    mu_g = p_return(w, mu1, mu2)
    sg_g = p_std(w, s1, s2, rho)
    eg_g = p_esg(w, e1, e2)
    ut_g = p_util(mu_g, sg_g, eg_g, gamma, lam)
    idx  = np.argmax(ut_g)
    return w, mu_g, sg_g, eg_g, ut_g, mu_g + lam * eg_g, idx


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌿 GreenVest  |  ESG Portfolio Optimiser")
st.caption("Build your optimal ESG portfolio. Adjust any input — results update instantly.")

tab1, tab2, tab3 = st.tabs(["📋  Inputs", "📊  Results", "📈  Charts"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Step 1: Risk Profile ───────────────────────────────────────────────────
    st.subheader("Step 1 · Risk Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        q1 = st.selectbox(
            "Portfolio drops 25% — you would:",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1:"Sell everything",2:"Reduce exposure",
                                    3:"Hold steady",4:"Buy more"}[x],
        )
    with col2:
        q2 = st.selectbox(
            "Your main goal:",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1:"Preserve capital",2:"Generate income",
                                    3:"Long-term growth",4:"Maximise growth"}[x],
        )
    with col3:
        q3 = st.selectbox(
            "Investment horizon:",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1:"Less than 2 years",2:"2–5 years",
                                    3:"5–10 years",4:"10+ years"}[x],
        )

    score      = ((5 - q1) + (5 - q2) + q3) / 3
    gamma_base = round(2 + (score - 1) * (8 / 3), 1)
    profile    = ("Conservative" if gamma_base >= 7
                  else "Balanced" if gamma_base >= 4
                  else "Aggressive")
    st.info(f"**Risk Profile: {profile}** · γ = {gamma_base}")

    st.divider()

    # ── Step 2: Investment Goal ────────────────────────────────────────────────
    st.subheader("Step 2 · Investment Goal")
    goal = st.selectbox(
        "What are you investing for?",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1:"Retirement",2:"Growth",
                                3:"Ethical investing",4:"Short-term profit"}[x],
    )
    gamma    = round(max(1.0, gamma_base + {1:+2, 2:0, 3:0, 4:-1}[goal]), 1)
    goal_lam = {1:0, 2:0, 3:0.15, 4:0}[goal]

    st.divider()

    # ── Step 3: ESG Priorities ─────────────────────────────────────────────────
    st.subheader("Step 3 · ESG Priorities")
    st.caption("0 = don't care · 5 = essential")
    col1, col2, col3 = st.columns(3)
    with col1:
        e_w = st.slider("🌍 Environmental", 0, 5, 3)
    with col2:
        s_w = st.slider("🤝 Social", 0, 5, 3)
    with col3:
        g_w = st.slider("🏛 Governance", 0, 5, 3)

    st.caption("Sector exclusions")
    excl_cols   = st.columns(5)
    excl_count  = 0
    for i, sector in enumerate(["Tobacco", "Weapons", "Gambling", "Fossil Fuels", "Alcohol"]):
        if excl_cols[i].checkbox(sector):
            excl_count += 1

    lam_base   = round(((e_w + s_w + g_w) / 3 / 5) * 0.85 + 0.05, 3)
    lambda_esg = round(min(lam_base + 0.03 * excl_count + goal_lam, 1.0), 3)
    st.info(f"**λ (ESG weight) = {lambda_esg}** · γ (risk aversion) = {gamma}")

    st.divider()

    # ── Step 4: Market Parameters ──────────────────────────────────────────────
    st.subheader("Step 4 · Market Parameters")
    col1, col2 = st.columns(2)
    with col1:
        rf = st.number_input(
            "Risk-free rate (%)", min_value=0.0, max_value=20.0,
            value=4.5, step=0.1,
        ) / 100
    with col2:
        invest = st.number_input(
            "Investment amount (GBP)", min_value=100.0, max_value=1_000_000.0,
            value=10_000.0, step=500.0,
        )

    st.divider()

    # ── Step 5 & 6: Asset Inputs ───────────────────────────────────────────────
    assets = {}
    for i in [1, 2]:
        st.subheader(f"Step {4+i} · Asset {i}")
        col_main, col_esg, col_hist = st.columns([1, 1, 1])

        with col_main:
            name  = st.text_input("Name", value=f"Asset {i}", key=f"name{i}")
            mu    = st.number_input(
                "Expected Return (%)", min_value=-100.0, max_value=500.0,
                value=8.0 if i == 1 else 5.0, step=0.1, key=f"mu{i}",
            ) / 100
            sigma = st.number_input(
                "Std Deviation (%)", min_value=0.01, max_value=500.0,
                value=15.0 if i == 1 else 10.0, step=0.1, key=f"sigma{i}",
            ) / 100

        with col_esg:
            st.caption("ESG Sub-scores (0–100)")
            e_score = st.slider("Environmental", 0.0, 100.0,
                                70.0 if i == 1 else 50.0, key=f"e{i}")
            s_score = st.slider("Social",        0.0, 100.0,
                                65.0 if i == 1 else 55.0, key=f"s{i}")
            g_score = st.slider("Governance",    0.0, 100.0,
                                60.0 if i == 1 else 45.0, key=f"g{i}")

        with col_hist:
            esg_last = st.number_input(
                "Last Year's ESG Score (0–100)", min_value=0.0, max_value=100.0,
                value=65.0 if i == 1 else 48.0, key=f"esg_last{i}",
            )

        esg_c = composite_esg(e_score, s_score, g_score, e_w, s_w, g_w) / 100
        mom   = esg_momentum(
            composite_esg(e_score, s_score, g_score, e_w, s_w, g_w), esg_last
        )
        esg_a      = min(esg_c + 0.05 * mom, 1.0)
        greenwash  = esg_c >= 0.60 and g_score / 100 < 0.35

        st.caption(
            f"Composite ESG: **{esg_c*100:.1f}** [{esg_rating(esg_c)}] · "
            f"Momentum: {'+' if mom >= 0 else ''}{mom*100:.1f}% · "
            f"Adjusted: **{esg_a*100:.1f}**"
        )
        if greenwash:
            st.warning(f"⚠️ Greenwashing Alert: High ESG but low Governance on {name}.")

        assets[i] = dict(
            name=name, mu=mu, sigma=sigma,
            e=e_score, s=s_score, g=g_score,
            esg_last=esg_last, esg_c=esg_c, mom=mom, esg_a=esg_a,
        )

        if i == 1:
            st.divider()

    st.divider()
    rho = st.slider(
        "Correlation between assets (ρ)", -1.0, 1.0, 0.3, step=0.01,
        help="How much the two assets move together. -1 = perfect inverse, 0 = uncorrelated, 1 = perfect positive.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE — runs on every widget change (reactive)
# ══════════════════════════════════════════════════════════════════════════════
a1, a2 = assets[1], assets[2]

w, mu_g, sg_g, eg_g, ut_g, ma_g, idx = optimise(
    a1["mu"], a2["mu"], a1["sigma"], a2["sigma"],
    a1["esg_a"], a2["esg_a"], rho, gamma, lambda_esg,
)

w1   = w[idx];  w2   = 1 - w1
mu_o = mu_g[idx]; sg_o = sg_g[idx]
eg_o = eg_g[idx]; ut_o = ut_g[idx]
sh_o    = (mu_o - rf) / sg_o if sg_o > 0 else float("nan")
esg_sh  = esg_sharpe(mu_o, rf, sg_o, lambda_esg, eg_o)
imp     = round(eg_o * 100, 1)
td      = lambda_esg * eg_o - gamma * sg_o

_, mu_ms, sg_ms, eg_ms, _, _, idx_ms = optimise(
    a1["mu"], a2["mu"], a1["sigma"], a2["sigma"],
    a1["esg_a"], a2["esg_a"], rho, gamma, 0.0,
)
idx_mr   = np.argmin(sg_g)
idx_me   = np.argmax(eg_g)
mu50     = p_return(0.5, a1["mu"], a2["mu"])
sg50     = p_std(0.5, a1["sigma"], a2["sigma"], rho)
eg50     = p_esg(0.5, a1["esg_a"], a2["esg_a"])
ret_cost = mu_ms[idx_ms] * 100 - mu_o * 100


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Optimal Portfolio")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{a1['name']} weight",  f"{w1*100:.2f}%")
    col2.metric(f"{a2['name']} weight",  f"{w2*100:.2f}%")
    col3.metric("Expected Return",        f"{mu_o*100:.2f}%")
    col4.metric("Std Deviation",          f"{sg_o*100:.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio",           f"{sh_o:.4f}")
    col2.metric(f"ESG Score [{esg_rating(eg_o)}]", f"{eg_o*100:.1f} / 100")
    col3.metric("ESG-Adjusted Sharpe",    f"{esg_sh:.4f}")
    col4.metric("GreenVest Impact Score", f"{imp:.1f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trade-off Score",  f"{td:.4f}")
    col2.metric("Utility Value",    f"{ut_o:.6f}")
    col3.metric("γ (risk aversion)",f"{gamma}")
    col4.metric("λ (ESG weight)",   f"{lambda_esg}")

    if ret_cost > 1.0:
        st.warning(
            f"⚠️ **Regret Warning:** Your ESG tilt costs {ret_cost:.2f}% "
            f"vs the financial-only optimal."
        )
    elif ret_cost < 0:
        st.success(
            f"✅ **ESG pays off:** Your portfolio beats the financial-only "
            f"optimal by {abs(ret_cost):.2f}%."
        )

    st.divider()

    # ── ESG Breakdown ──────────────────────────────────────────────────────────
    st.subheader("ESG Breakdown")
    esg_df = pd.DataFrame({
        "Pillar":       ["Environmental","Social","Governance","Composite","Momentum-adj"],
        a1["name"]:     [a1["e"], a1["s"], a1["g"],
                         f"{a1['esg_c']*100:.1f}", f"{a1['esg_a']*100:.1f}"],
        a2["name"]:     [a2["e"], a2["s"], a2["g"],
                         f"{a2['esg_c']*100:.1f}", f"{a2['esg_a']*100:.1f}"],
        "Your Weight":  [f"{e_w}/5", f"{s_w}/5", f"{g_w}/5", "—", "—"],
    })
    st.dataframe(esg_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Portfolio Comparison ───────────────────────────────────────────────────
    st.subheader("Portfolio Comparison")
    comp_df = pd.DataFrame({
        "Portfolio": ["GreenVest Optimal", "Max Sharpe (λ=0)",
                      "Min Risk", "Max ESG", "50 / 50"],
        f"W({a1['name'][:10]})": [
            f"{w1*100:.1f}%", f"{w[idx_ms]*100:.1f}%",
            f"{w[idx_mr]*100:.1f}%", f"{w[idx_me]*100:.1f}%", "50.0%",
        ],
        "Return": [
            f"{mu_o*100:.2f}%",         f"{mu_ms[idx_ms]*100:.2f}%",
            f"{mu_g[idx_mr]*100:.2f}%", f"{mu_g[idx_me]*100:.2f}%",
            f"{mu50*100:.2f}%",
        ],
        "Std Dev": [
            f"{sg_o*100:.2f}%",         f"{sg_ms[idx_ms]*100:.2f}%",
            f"{sg_g[idx_mr]*100:.2f}%", f"{sg_g[idx_me]*100:.2f}%",
            f"{sg50*100:.2f}%",
        ],
        "ESG Score": [
            f"{eg_o*100:.1f}", f"{eg_ms[idx_ms]*100:.1f}",
            f"{eg_g[idx_mr]*100:.1f}", f"{eg_g[idx_me]*100:.1f}",
            f"{eg50*100:.1f}",
        ],
        "Rating": [
            esg_rating(eg_o),       esg_rating(eg_ms[idx_ms]),
            esg_rating(eg_g[idx_mr]), esg_rating(eg_g[idx_me]),
            esg_rating(eg50),
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Future Value ───────────────────────────────────────────────────────────
    st.subheader(f"Future Value  ·  GBP {invest:,.0f} invested")
    fv_df = pd.DataFrame({
        "Horizon":        ["5 years", "10 years", "20 years", "30 years"],
        "GreenVest":      [f"GBP {future_value(invest, mu_o, y):,.0f}"  for y in [5,10,20,30]],
        "50 / 50":        [f"GBP {future_value(invest, mu50, y):,.0f}"  for y in [5,10,20,30]],
        "Difference":     [
            f"GBP {future_value(invest,mu_o,y)-future_value(invest,mu50,y):+,.0f}"
            for y in [5,10,20,30]
        ],
    })
    st.dataframe(fv_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Sensitivity Table ──────────────────────────────────────────────────────
    st.subheader("Sensitivity  ·  Optimal allocation as λ changes")
    lam_vals  = np.linspace(0, 1, 11)
    sens_rows = []
    for lam_i in lam_vals:
        _, mi, si, ei, _, _, xi = optimise(
            a1["mu"], a2["mu"], a1["sigma"], a2["sigma"],
            a1["esg_a"], a2["esg_a"], rho, gamma, lam_i,
        )
        sens_rows.append({
            "λ":                   f"{lam_i:.1f}",
            f"W({a1['name'][:10]})": f"{w[xi]*100:.1f}%",
            "Return":              f"{mi[xi]*100:.2f}%",
            "ESG Score":           f"{ei[xi]*100:.1f}",
            "Rating":              esg_rating(ei[xi]),
        })
    st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("GreenVest  —  ESG Portfolio Analysis",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#f7f4ef")

    def style(ax):
        ax.set_facecolor("#f7f4ef")
        ax.grid(True, alpha=0.25)

    # 1. ESG-coloured frontier ─────────────────────────────────────────────────
    ax = axes[0][0]; style(ax)
    pts  = np.array([sg_g, ma_g]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="YlGn", linewidth=2.5)
    lc.set_array(eg_g)
    ax.add_collection(lc)
    ax.autoscale()
    ax.plot(sg_g, mu_g, color="#ccc", linewidth=1.2, linestyle="--",
            label="Traditional frontier")
    fig.colorbar(lc, ax=ax, shrink=0.8, label="ESG Score")
    ax.scatter(sg_o,          ma_g[idx],     color="#0f2d1e", s=100,
               marker="D", label="GreenVest Opt", zorder=5, edgecolors="#7ec8a0")
    ax.scatter(sg_ms[idx_ms], ma_g[idx_ms],  color="#b87333", s=60,
               marker="^",  label="Max Sharpe",   zorder=4)
    ax.scatter(sg_g[idx_mr],  ma_g[idx_mr],  color="#3a5fb8", s=60,
               marker="s",  label="Min Risk",     zorder=4)
    ax.set_xlabel("Std Dev"); ax.set_ylabel("ESG-Adjusted Return")
    ax.set_title("ESG-Efficient Frontier"); ax.legend(fontsize=7)

    # 2. Utility curve ─────────────────────────────────────────────────────────
    ax = axes[0][1]; style(ax)
    ax.plot(w * 100, ut_g, color="#3a6b4a", linewidth=2)
    ax.axvline(w1 * 100, color="#0f2d1e", linestyle="--", alpha=0.7)
    ax.scatter(w1 * 100, ut_o, color="#0f2d1e", s=90, marker="D", zorder=5)
    ax.set_xlabel(f"Weight in {a1['name']} (%)")
    ax.set_ylabel("Utility"); ax.set_title("Utility vs Portfolio Weight")

    # 3. Future value ──────────────────────────────────────────────────────────
    ax = axes[0][2]; style(ax)
    yr = np.arange(0, 31)
    ax.plot(yr, [future_value(invest, mu_o,  y) for y in yr],
            color="#0f2d1e", linewidth=2.5,
            label=f"GreenVest ({mu_o*100:.1f}%)")
    ax.plot(yr, [future_value(invest, mu50,  y) for y in yr],
            color="#b87333", linewidth=1.8, linestyle="--",
            label=f"50/50 ({mu50*100:.1f}%)")
    ax.plot(yr, [future_value(invest, rf,    y) for y in yr],
            color="#aaa",    linewidth=1.2, linestyle=":",
            label=f"Risk-free ({rf*100:.1f}%)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"GBP{x/1000:.0f}k")
    )
    ax.set_xlabel("Years"); ax.set_title("Future Value"); ax.legend(fontsize=7)

    # 4–6. Sensitivity charts ──────────────────────────────────────────────────
    lam_r = np.linspace(0, 1, 60)
    w1_s, mu_s, esg_s = [], [], []
    for lam_i in lam_r:
        _, mi, si, ei, _, _, xi = optimise(
            a1["mu"], a2["mu"], a1["sigma"], a2["sigma"],
            a1["esg_a"], a2["esg_a"], rho, gamma, lam_i,
        )
        w1_s.append(w[xi] * 100)
        mu_s.append(mi[xi] * 100)
        esg_s.append(ei[xi] * 100)

    for ax, data, ylabel, color, title in [
        (axes[1][0], w1_s,  f"W({a1['name'][:10]}) %", "#3a6b4a", "Sensitivity: Allocation"),
        (axes[1][1], mu_s,  "Return (%)",               "#b87333", "Sensitivity: Return"),
        (axes[1][2], esg_s, "ESG Score",                "#0f2d1e", "Sensitivity: ESG"),
    ]:
        style(ax)
        ax.plot(lam_r, data, color=color, linewidth=2)
        ax.axvline(lambda_esg, color="#888", linestyle="--", alpha=0.7,
                   label=f"Your λ = {lambda_esg}")
        ax.set_xlabel("ESG Weight (λ)")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
