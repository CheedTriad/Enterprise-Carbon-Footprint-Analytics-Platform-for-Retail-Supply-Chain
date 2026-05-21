"""
app.py  --  GreenMart Carbon Footprint Analytics Platform
Run:  py app.py
Opens at: http://localhost:8050
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx, no_update, ALL
import dash_bootstrap_components as dbc
import uuid
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
products = pd.read_csv("data/fb_products_clean.csv")
stages   = pd.read_csv("data/fb_stages_clean.csv")

# ── GreenMart anchor companies ────────────────────────────────────────────────
ANCHOR_COMPANIES = [
    "Coca-Cola Enterprises, Inc.",
    "Barilla Holding SpA",
    "Nestle",
    "Danone",
    "CJ Cheiljedang",
]

ALL_COMPANIES = ["All Companies"] + sorted(products["company"].unique().tolist())

# ── Colour palette (light neutral theme) ──────────────────────────────────────
PAGE_BG    = "#F5F3EF"   # warm off-white / beige
CARD_BG    = "#FFFFFF"   # white cards
SIDEBAR_BG = "#2C3E50"   # dark blue-grey sidebar
TEXT_DARK  = "#1C1C1C"   # near-black body text
TEXT_MID   = "#5A5A5A"   # secondary text
TEXT_LIGHT = "#ECEFF1"   # sidebar text
BORDER     = "#DDD8D0"   # warm grey border
ACCENT     = "#3B6EA5"   # steel blue accent
HOTSPOT    = "#B5451B"   # terracotta for hotspot
SUCCESS    = "#3A7D44"   # forest green for positive
CUSTOM     = "#00838F"   # teal for user-submitted "Your Company"

CHART_PALETTE = [
    "#3B6EA5",  # steel blue
    "#B5451B",  # terracotta
    "#3A7D44",  # forest green
    "#C9943A",  # amber
    "#6B4C8A",  # muted purple
    "#2E8B8B",  # teal
    "#8B4513",  # saddle brown
    "#5A7A3A",  # olive
]

STAGE_COLOURS = {
    "Upstream":    "#3B6EA5",
    "Operations":  "#C9943A",
    "Downstream":  "#B5451B",
    "Transport":   "#3A7D44",
    "End of Life": "#6B4C8A",
}

# ── Glossary definitions ───────────────────────────────────────────────────────
GLOSSARY = {
    "pcf": (
        "Product Carbon Footprint (PCF)",
        "Total greenhouse gas emissions from a product across its full life cycle — "
        "from raw material extraction through end-of-life disposal. Measured in kg CO₂e.",
    ),
    "carbon_intensity": (
        "Carbon Intensity",
        "Emissions per unit mass of product (kg CO₂e / kg). Enables fair comparison "
        "between products of different sizes or across companies.",
    ),
    "co2e": (
        "CO₂e — Carbon Dioxide Equivalent",
        "A unit expressing the combined warming impact of all greenhouse gases (CO₂, "
        "CH₄, N₂O, etc.) relative to CO₂ over a 100-year horizon.",
    ),
    "upstream": (
        "Upstream Emissions",
        "Emissions from raw material extraction, agriculture, ingredient production, "
        "and supplier operations — everything before the manufacturer's own gate.",
    ),
    "downstream": (
        "Downstream Emissions",
        "Emissions from distribution, retail, consumer use, and product disposal "
        "after the manufacturing gate.",
    ),
    "operations": (
        "Operations / Direct Emissions",
        "Emissions directly from the manufacturer's own facilities and processes "
        "(analogous to Scope 1 + 2 in the GHG Protocol).",
    ),
    "transport": (
        "Transport Emissions",
        "Emissions from moving goods between supply chain stages, including inbound "
        "raw-material logistics and outbound distribution.",
    ),
    "end_of_life": (
        "End-of-Life Emissions",
        "Emissions from disposing of packaging and food waste, including landfill, "
        "composting, incineration, and recycling.",
    ),
    "lca": (
        "Life Cycle Assessment (LCA)",
        "A methodology for quantifying all environmental impacts across a product's "
        "full life — from raw material extraction ('cradle') to final disposal ('grave').",
    ),
    "kmeans": (
        "K-Means Clustering",
        "An unsupervised ML algorithm that groups products into k clusters based on "
        "similarity of their emission-stage fractions. The cluster with the highest "
        "combined upstream + operations fraction is flagged as the hotspot.",
    ),
    "hotspot": (
        "Emission Hotspot",
        "A product or cluster with disproportionately high emissions, identified here "
        "by K-Means as the cluster with the highest combined upstream + operations fraction.",
    ),
    "roi_score": (
        "ROI Score",
        "A composite score (0–1) for each intervention representing CO₂e reduction "
        "per unit of cost. Calculated as tonnes saved ÷ GBP cost, then normalised. "
        "Higher = more cost-effective.",
    ),
    "payback": (
        "Payback Period",
        "Estimated years to recover the intervention cost through avoided carbon costs, "
        "calculated at a shadow carbon price of £50 / tonne CO₂e.",
    ),
    "emission_fraction": (
        "Emission Fraction",
        "The share of a product's total PCF attributed to one life-cycle stage "
        "(e.g. upstream fraction = upstream emissions ÷ total PCF). All stage fractions sum to 1.",
    ),
    "hvo": (
        "HVO — Hydrotreated Vegetable Oil",
        "A renewable diesel substitute produced from biological waste streams, with "
        "lifecycle emissions typically 70–90 % lower than fossil diesel.",
    ),
    "value_chain": (
        "Value Chain",
        "The full sequence of activities — raw materials → manufacturing → logistics "
        "→ retail → consumer → disposal — that create and deliver a product.",
    ),
    "scope3": (
        "Scope 1 / 2 / 3 Emissions",
        "GHG Protocol categories: Scope 1 = direct combustion; Scope 2 = purchased "
        "energy; Scope 3 = all other value-chain emissions — the largest category for "
        "most F&B companies, covering upstream agriculture and downstream logistics.",
    ),
}

# ── ML: K-Means hotspot detection ─────────────────────────────────────────────
def run_kmeans(df, n_clusters=4):
    features = ["upstream_frac", "ops_frac", "downstream_frac", "transport_frac"]
    X = df[features].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df = df.copy()
    df["cluster_id"] = labels
    cluster_means = pd.DataFrame(X_imp, columns=features)
    cluster_means["cluster_id"] = labels
    cluster_intensity = cluster_means.groupby("cluster_id")[["upstream_frac","ops_frac"]].mean().sum(axis=1)
    hotspot_cluster = int(cluster_intensity.idxmax())
    df["is_hotspot"] = df["cluster_id"] == hotspot_cluster
    return df

products = run_kmeans(products)

# ── Cost-benefit interventions (GreenMart scenario, synthetic) ────────────────
INTERVENTIONS = pd.DataFrame([
    {"stage": "Upstream",    "description": "Switch to certified sustainable ingredient sourcing",     "est_co2e_saving_tonne": 42.5, "est_cost_gbp": 85000,  "roi_score": 0.85},
    {"stage": "Transport",   "description": "Transition primary logistics fleet to HVO fuel",           "est_co2e_saving_tonne": 31.2, "est_cost_gbp": 62000,  "roi_score": 0.78},
    {"stage": "Operations",  "description": "Install on-site solar PV at 3 manufacturing sites",       "est_co2e_saving_tonne": 28.0, "est_cost_gbp": 140000, "roi_score": 0.62},
    {"stage": "Upstream",    "description": "Reduce agricultural water use via precision irrigation",   "est_co2e_saving_tonne": 19.8, "est_cost_gbp": 38000,  "roi_score": 0.91},
    {"stage": "Downstream",  "description": "Optimise cold-chain packaging to reduce refrigeration",    "est_co2e_saving_tonne": 15.4, "est_cost_gbp": 27000,  "roi_score": 0.88},
    {"stage": "Operations",  "description": "Replace gas boilers with heat pump systems",               "est_co2e_saving_tonne": 22.1, "est_cost_gbp": 95000,  "roi_score": 0.70},
    {"stage": "Transport",   "description": "Consolidate inbound delivery routes (route optimisation)", "est_co2e_saving_tonne": 11.6, "est_cost_gbp": 14000,  "roi_score": 0.94},
    {"stage": "End of Life", "description": "Partner with waste processor for food waste anaerobic digestion", "est_co2e_saving_tonne": 8.3, "est_cost_gbp": 22000, "roi_score": 0.72},
])
INTERVENTIONS["payback_years"] = (INTERVENTIONS["est_cost_gbp"] / (INTERVENTIONS["est_co2e_saving_tonne"] * 50)).round(1)
INTERVENTIONS = INTERVENTIONS.sort_values("roi_score", ascending=False)

# ── Helpers ───────────────────────────────────────────────────────────────────
def filter_products(company):
    if company in ("All Companies", "__new__", None):
        return products
    return products[products["company"] == company]

def chart_layout(fig, height=400):
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT_DARK,
        font_family="Inter, Helvetica Neue, sans-serif",
        title_font_size=13, title_font_color=TEXT_DARK,
        height=height,
        margin=dict(t=48, b=32, l=16, r=16),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER)
    return fig

def kpi_card(title, value, subtitle="", accent=ACCENT, tooltip_key=None):
    title_el = with_tooltip(title, tooltip_key) if tooltip_key else title
    return dbc.Card([
        dbc.CardBody([
            html.P(title_el, style={
                "fontSize": "0.7rem", "textTransform": "uppercase",
                "letterSpacing": "0.07em", "color": TEXT_MID,
                "marginBottom": "6px", "fontWeight": "600",
            }),
            html.H3(value, style={
                "color": accent, "fontWeight": "700",
                "marginBottom": "4px", "fontSize": "1.5rem",
            }),
            html.P(subtitle, style={"fontSize": "0.74rem", "color": TEXT_MID, "marginBottom": "0"}),
        ], style={"padding": "14px 16px"}),
    ], style={
        "backgroundColor": CARD_BG,
        "borderTop": f"3px solid {accent}",
        "border": f"1px solid {BORDER}",
        "borderRadius": "6px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
    })

def section_header(title, subtitle=None):
    return html.Div([
        html.H5(title, style={"color": TEXT_DARK, "fontWeight": "700", "marginBottom": "2px"}),
        html.P(subtitle or "", style={"color": TEXT_MID, "fontSize": "0.82rem", "marginBottom": "0"}),
    ], style={"marginBottom": "20px", "borderBottom": f"1px solid {BORDER}", "paddingBottom": "12px"})

def card_wrap(content):
    return dbc.Card(content, style={
        "backgroundColor": CARD_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "6px", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
        "overflow": "hidden",
    })

def with_tooltip(text, key):
    """Wrap a term in a span with a hover tooltip sourced from GLOSSARY[key]."""
    tid = f"tt-{key}-{uuid.uuid4().hex[:8]}"
    return html.Span([
        html.Span(text, id=tid, style={
            "cursor": "help",
            "borderBottom": f"1px dotted {TEXT_MID}",
            "display": "inline",
        }),
        dbc.Tooltip(
            [html.Strong(GLOSSARY[key][0]), html.Br(), GLOSSARY[key][1]],
            target=tid,
            placement="top",
            style={"maxWidth": "300px", "fontSize": "0.78rem", "lineHeight": "1.5", "textAlign": "left"},
        ),
    ])

def make_glossary_offcanvas():
    items = []
    for _, (name, defn) in sorted(GLOSSARY.items(), key=lambda x: x[1][0]):
        items.append(html.Div([
            html.Dt(name, style={
                "fontWeight": "700", "color": TEXT_DARK,
                "fontSize": "0.83rem", "marginBottom": "3px",
            }),
            html.Dd(defn, style={
                "color": TEXT_MID, "fontSize": "0.79rem",
                "lineHeight": "1.5", "marginBottom": "14px", "marginLeft": "0",
            }),
            html.Hr(style={"borderColor": BORDER, "margin": "0 0 14px"}),
        ]))
    return dbc.Offcanvas(
        html.Div([
            html.P(
                "Hover over any underlined term in the dashboard for a quick definition. "
                "Full glossary below.",
                style={
                    "color": TEXT_MID, "fontSize": "0.8rem", "marginBottom": "18px",
                    "borderBottom": f"1px solid {BORDER}", "paddingBottom": "12px",
                },
            ),
            html.Dl(items, style={"margin": "0"}),
        ], style={"padding": "4px"}),
        id="glossary-offcanvas",
        title="Glossary",
        placement="end",
        is_open=False,
        style={"width": "390px", "fontFamily": "Inter, sans-serif"},
    )

def make_new_company_modal():
    lbl = {"fontSize": "0.72rem", "color": TEXT_MID, "marginBottom": "3px",
           "fontWeight": "600", "display": "block"}
    inp = {"fontSize": "0.82rem", "height": "32px"}
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Add Your Company to the Benchmark")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Company name *", style=lbl),
                    dbc.Input(id="nc-name", placeholder="e.g. Acme Foods Ltd",
                              type="text", style=inp),
                ], width=6),
                dbc.Col([
                    html.Label("Product name *", style=lbl),
                    dbc.Input(id="nc-product", placeholder="e.g. Granola Bar 50g",
                              type="text", style=inp),
                ], width=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Total PCF (kg CO₂e) *", style=lbl),
                    dbc.Input(id="nc-pcf", placeholder="e.g. 3.75",
                              type="number", min=0, style=inp),
                ], width=6),
                dbc.Col([
                    html.Label("Product weight (kg) *", style=lbl),
                    dbc.Input(id="nc-weight", placeholder="e.g. 0.5",
                              type="number", min=0.001, step=0.001, style=inp),
                ], width=6),
            ], className="mb-3"),
            html.Hr(style={"borderColor": BORDER, "margin": "4px 0 12px"}),
            html.P("Stage breakdown — must sum to 100 %", style={
                "fontSize": "0.72rem", "color": TEXT_MID,
                "fontWeight": "600", "marginBottom": "10px",
            }),
            dbc.Row([
                dbc.Col([
                    html.Label("Upstream %", style=lbl),
                    dbc.Input(id="nc-upstream", type="number", min=0, max=100,
                              placeholder="e.g. 60", style=inp),
                ], width=4),
                dbc.Col([
                    html.Label("Operations %", style=lbl),
                    dbc.Input(id="nc-ops", type="number", min=0, max=100,
                              placeholder="e.g. 15", style=inp),
                ], width=4),
                dbc.Col([
                    html.Label("Downstream %", style=lbl),
                    dbc.Input(id="nc-downstream", type="number", min=0, max=100,
                              placeholder="e.g. 10", style=inp),
                ], width=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("Transport %", style=lbl),
                    dbc.Input(id="nc-transport", type="number", min=0, max=100,
                              placeholder="e.g. 10", style=inp),
                ], width=4),
                dbc.Col([
                    html.Label("End of Life %", style=lbl),
                    dbc.Input(id="nc-eol", type="number", min=0, max=100,
                              placeholder="e.g. 5", style=inp),
                ], width=4),
                dbc.Col([
                    html.Label(" ", style={**lbl, "visibility": "hidden"}),
                    html.Div(id="nc-stage-sum", children="Total: 0 %", style={
                        "fontSize": "0.82rem", "fontWeight": "700",
                        "color": TEXT_MID, "lineHeight": "32px",
                    }),
                ], width=4),
            ], className="mb-3"),
            html.Div(id="nc-error", style={
                "color": HOTSPOT, "fontSize": "0.8rem", "minHeight": "20px",
            }),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="nc-cancel", color="secondary",
                       size="sm", className="me-2", n_clicks=0),
            dbc.Button("Add to Benchmark", id="nc-submit", color="primary",
                       size="sm", n_clicks=0),
        ]),
    ], id="new-company-modal", is_open=False, size="lg")

def table_styles(header_colour=TEXT_DARK):
    return dict(
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": CARD_BG, "color": TEXT_DARK,
            "border": f"1px solid {BORDER}", "fontSize": "12px",
            "padding": "8px 10px", "fontFamily": "Inter, sans-serif",
        },
        style_header={
            "backgroundColor": "#F0EDE8", "color": header_colour,
            "fontWeight": "700", "border": f"1px solid {BORDER}",
            "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.05em",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"}],
    )

# ── Mobile sidebar (offcanvas) ────────────────────────────────────────────────
def make_mobile_offcanvas():
    return dbc.Offcanvas(
        html.Div([
            html.Div([
                html.Span("🌿", style={"fontSize": "1.3rem"}),
                html.Span(" GreenMart", style={"fontSize": "1.05rem", "fontWeight": "800", "color": TEXT_LIGHT, "marginLeft": "6px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
            html.P("Carbon Analytics Platform", style={"color": "#8FA8BF", "fontSize": "0.73rem", "marginBottom": "20px"}),

            html.Label("Company Filter", style={"color": "#8FA8BF", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.07em", "marginBottom": "6px", "display": "block"}),
            dcc.Dropdown(
                id="company-filter-mobile",
                options=(
                    [{"label": "➕  Add your company...", "value": "__new__"}] +
                    [{"label": c, "value": c} for c in ALL_COMPANIES]
                ),
                value="All Companies",
                clearable=False,
                style={"marginBottom": "8px", "fontSize": "12px"},
            ),

            html.Hr(style={"borderColor": "#3D5166", "marginBottom": "14px"}),
            html.P("Navigation", style={"color": "#8FA8BF", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.07em", "marginBottom": "8px"}),
            dbc.Nav([
                dbc.NavLink("📊  Overview",          href="/",             active="exact"),
                dbc.NavLink("🔬  Stage Breakdown",   href="/stages",       active="exact"),
                dbc.NavLink("🎯  Hotspot Detection", href="/hotspots",     active="exact"),
                dbc.NavLink("🏢  Benchmarking",      href="/benchmarking", active="exact"),
                dbc.NavLink("💡  Interventions",     href="/interventions",active="exact"),
            ], vertical=True, pills=True),

            html.Hr(style={"borderColor": "#3D5166", "marginTop": "14px", "marginBottom": "10px"}),
            html.P("Reference", style={"color": "#8FA8BF", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.07em", "marginBottom": "8px"}),
            html.Button(
                "📖  Glossary",
                id="open-glossary-btn-mobile",
                n_clicks=0,
                style={
                    "background": "none", "border": "none", "color": TEXT_LIGHT,
                    "borderRadius": "0.25rem", "padding": "0.5rem 1rem",
                    "fontSize": "0.875rem", "cursor": "pointer",
                    "width": "100%", "textAlign": "left", "display": "block", "opacity": "0.85",
                },
            ),

            html.Hr(style={"borderColor": "#3D5166", "marginTop": "14px", "marginBottom": "10px"}),
            html.P("Data: Carbon Catalogue", style={"color": "#546E7A", "fontSize": "0.67rem", "marginBottom": "2px"}),
            html.P("Kaack et al., 2022 · CC BY 4.0", style={"color": "#546E7A", "fontSize": "0.67rem", "marginBottom": "2px"}),
            html.P("F&B Sector · 139 products · 351 stages", style={"color": "#546E7A", "fontSize": "0.67rem"}),
        ], style={"padding": "22px 18px"}),
        id="mobile-sidebar-offcanvas",
        is_open=False,
        placement="start",
        style={"width": "285px", "backgroundColor": SIDEBAR_BG},
        title=None,
        backdrop=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR = dbc.Col([
    html.Div([
        html.Div([
            html.Span("🌿", style={"fontSize": "1.3rem"}),
            html.Span(" GreenMart", style={"fontSize": "1.05rem", "fontWeight": "800", "color": TEXT_LIGHT, "marginLeft": "6px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
        html.P("Carbon Analytics Platform", style={"color": "#8FA8BF", "fontSize": "0.73rem", "marginBottom": "28px"}),

        html.Label("Company Filter", style={"color": "#8FA8BF", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.07em", "marginBottom": "6px", "display": "block"}),
        dcc.Dropdown(
            id="company-filter",
            options=(
                [{"label": "➕  Add your company...", "value": "__new__"}] +
                [{"label": c, "value": c} for c in ALL_COMPANIES]
            ),
            value="All Companies",
            clearable=False,
            style={"marginBottom": "8px", "fontSize": "12px"},
        ),
        html.Div(id="nc-active-banner", style={"display": "none", "marginBottom": "16px"}),

        html.Hr(style={"borderColor": "#3D5166", "marginBottom": "14px"}),
        html.P("Navigation", style={"color": "#8FA8BF", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.07em", "marginBottom": "8px"}),
        dbc.Nav([
            dbc.NavLink("📊  Overview",          href="/",             active="exact"),
            dbc.NavLink("🔬  Stage Breakdown",   href="/stages",       active="exact"),
            dbc.NavLink("🎯  Hotspot Detection", href="/hotspots",     active="exact"),
            dbc.NavLink("🏢  Benchmarking",      href="/benchmarking", active="exact"),
            dbc.NavLink("💡  Interventions",     href="/interventions",active="exact"),
        ], vertical=True, pills=True),

        html.Hr(style={"borderColor": "#3D5166", "marginTop": "14px", "marginBottom": "10px"}),
        html.P("Reference", style={"color": "#8FA8BF", "fontSize": "0.68rem", "textTransform": "uppercase", "letterSpacing": "0.07em", "marginBottom": "8px"}),
        html.Button(
            "📖  Glossary",
            id="open-glossary-btn",
            n_clicks=0,
            style={
                "background": "none",
                "border": "none",
                "color": TEXT_LIGHT,
                "borderRadius": "0.25rem",
                "padding": "0.5rem 1rem",
                "fontSize": "0.875rem",
                "cursor": "pointer",
                "width": "100%",
                "textAlign": "left",
                "display": "block",
                "opacity": "0.85",
            },
        ),

        html.Div([
            html.Hr(style={"borderColor": "#3D5166", "marginBottom": "10px"}),
            html.P("Data: Carbon Catalogue", style={"color": "#546E7A", "fontSize": "0.67rem", "marginBottom": "2px"}),
            html.P("Kaack et al., 2022 · CC BY 4.0", style={"color": "#546E7A", "fontSize": "0.67rem", "marginBottom": "2px"}),
            html.P("F&B Sector · 139 products · 351 stages", style={"color": "#546E7A", "fontSize": "0.67rem"}),
        ], style={"position": "absolute", "bottom": "20px", "left": "20px", "right": "20px"}),
    ], style={"padding": "22px 18px", "height": "100vh", "backgroundColor": SIDEBAR_BG, "position": "relative"})
], width=2, className="d-none d-md-block", style={"padding": "0"})

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "GreenMart | Carbon Analytics"
server = app.server  # expose Flask server for gunicorn

app.layout = dbc.Container([
    dcc.Location(id="url"),
    dcc.Store(id="custom-company-store", storage_type="session", data=None),
    dcc.Store(id="editing-company-store", data=None),
    make_glossary_offcanvas(),
    make_new_company_modal(),
    make_mobile_offcanvas(),
    # Mobile-only sticky top bar
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button(
                    "☰",
                    id="mobile-menu-btn",
                    color="link",
                    n_clicks=0,
                    style={"color": TEXT_LIGHT, "fontSize": "1.4rem", "padding": "0 8px", "lineHeight": "1"},
                ),
                html.Div([
                    html.Span("🌿", style={"fontSize": "1.1rem"}),
                    html.Span(" GreenMart", style={"fontSize": "1rem", "fontWeight": "800", "color": TEXT_LIGHT, "marginLeft": "4px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
        ], width=12),
    ], className="d-flex d-md-none mobile-topbar", style={"margin": "0"}),
    dbc.Row([
        SIDEBAR,
        dbc.Col([
            html.Div(id="page-content", style={"padding": "28px 32px"})
        ], xs=12, md=10, style={"backgroundColor": PAGE_BG, "minHeight": "100vh"}),
    ])
], fluid=True, style={"padding": "0", "fontFamily": "Inter, Helvetica Neue, sans-serif"})

# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview(company):
    df = filter_products(company)
    total_pcf            = df["pcf_kg_co2e"].sum()
    avg_intensity        = df["carbon_intensity"].mean()
    n_products           = len(df)
    n_hotspots           = int(df["is_hotspot"].sum())
    hotspot_intensity    = df[df["is_hotspot"]]["carbon_intensity"].mean()
    nonhotspot_intensity = df[~df["is_hotspot"]]["carbon_intensity"].mean()

    top15 = df.nlargest(15, "pcf_kg_co2e")[["product_name","pcf_kg_co2e","company"]].copy()
    top15["product_name"] = top15["product_name"].str[:50]
    fig_top = px.bar(
        top15, x="pcf_kg_co2e", y="product_name", orientation="h",
        color="company", color_discrete_sequence=CHART_PALETTE,
        labels={"pcf_kg_co2e": "PCF (kg CO2e)", "product_name": ""},
        title="Top 15 Products by Carbon Footprint",
    )
    chart_layout(fig_top, height=430)
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title_text="Company")

    fig_dist = px.histogram(
        df, x="pcf_kg_co2e", nbins=25,
        color_discrete_sequence=[ACCENT],
        labels={"pcf_kg_co2e": "PCF (kg CO2e)"},
        title="Distribution of Product Carbon Footprints",
    )
    chart_layout(fig_dist, height=290)

    return html.Div([
        section_header("Emissions Overview", "Portfolio-level summary across selected company filter"),
        dbc.Row([
            dbc.Col(kpi_card("Total PCF",             f"{total_pcf:,.0f} kg CO2e",       f"{n_products} products",                         tooltip_key="pcf"),              xs=6, md=2),
            dbc.Col(kpi_card("Avg Carbon Intensity",  f"{avg_intensity:.2f} kg/kg",       "per kg of product",                              tooltip_key="carbon_intensity"), xs=6, md=2),
            dbc.Col(kpi_card("Hotspot Products",      str(n_hotspots),                    "flagged by K-Means",      accent=HOTSPOT,         tooltip_key="hotspot"),          xs=6, md=2),
            dbc.Col(kpi_card("Portfolio Coverage",    f"{n_products}/139",                "F&B sector products"),                                                             xs=6, md=2),
            dbc.Col(kpi_card("Hotspot Intensity",     f"{hotspot_intensity:.2f} kg/kg",   "hotspot cluster mean",    accent=HOTSPOT,         tooltip_key="carbon_intensity"), xs=6, md=2),
            dbc.Col(kpi_card("Non-Hotspot Intensity", f"{nonhotspot_intensity:.2f} kg/kg","non-hotspot cluster mean",accent=SUCCESS,         tooltip_key="carbon_intensity"), xs=6, md=2),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(card_wrap(dcc.Graph(figure=fig_top,  config={"displayModeBar": False})), xs=12, md=8),
            dbc.Col(card_wrap(dcc.Graph(figure=fig_dist, config={"displayModeBar": False})), xs=12, md=4),
        ], className="g-3"),
    ])


def page_stages(company):
    df = filter_products(company)
    has_fracs = df.dropna(subset=["upstream_frac","ops_frac","downstream_frac"])

    mean_fracs = {
        "Upstream":    has_fracs["upstream_frac"].mean(),
        "Operations":  has_fracs["ops_frac"].mean(),
        "Downstream":  has_fracs["downstream_frac"].mean(),
        "Transport":   has_fracs["transport_frac"].mean(),
        "End of Life": has_fracs["endoflife_frac"].mean(),
    }
    fig_avg = go.Figure(go.Bar(
        x=list(mean_fracs.keys()),
        y=[v * 100 for v in mean_fracs.values()],
        marker_color=list(STAGE_COLOURS.values()),
        text=[f"{v*100:.1f}%" for v in mean_fracs.values()],
        textposition="outside",
    ))
    fig_avg.update_layout(title="Average Emission Fraction by Life Cycle Stage", xaxis_title="Stage", yaxis_title="% of Total PCF")
    chart_layout(fig_avg, height=340)

    top20 = has_fracs.nlargest(20, "pcf_kg_co2e").copy()
    top20["label"] = top20["product_name"].str[:38]
    fig_stack = go.Figure()
    for stage, col, colour in [
        ("Upstream",    "upstream_frac",   STAGE_COLOURS["Upstream"]),
        ("Operations",  "ops_frac",        STAGE_COLOURS["Operations"]),
        ("Downstream",  "downstream_frac", STAGE_COLOURS["Downstream"]),
        ("Transport",   "transport_frac",  STAGE_COLOURS["Transport"]),
        ("End of Life", "endoflife_frac",  STAGE_COLOURS["End of Life"]),
    ]:
        fig_stack.add_trace(go.Bar(name=stage, x=top20["label"], y=top20[col]*100, marker_color=colour))
    fig_stack.update_layout(barmode="stack", title="Stage Breakdown -- Top 20 Products by PCF",
                            xaxis_title="", yaxis_title="% of Total PCF",
                            xaxis_tickangle=-38, legend_title_text="Stage")
    chart_layout(fig_stack, height=420)

    stage_ids  = set(df["pcf_id"])
    stage_data = stages[stages["pcf_id"].isin(stage_ids)].copy()
    stage_data = stage_data.merge(df[["pcf_id","product_name","company"]], on="pcf_id", how="left")
    stage_data["product_name"] = stage_data["product_name"].str[:45]
    stage_data["stage_emissions_kg_co2e"] = stage_data["stage_emissions_kg_co2e"].round(3)

    tbl = dash_table.DataTable(
        data=stage_data[["product_name","company","stage_desc","value_chain","stage_emissions_kg_co2e"]].to_dict("records"),
        columns=[
            {"name": "Product",            "id": "product_name"},
            {"name": "Company",            "id": "company"},
            {"name": "LCA Stage",          "id": "stage_desc"},
            {"name": "Value Chain",        "id": "value_chain"},
            {"name": "Emissions (kg CO2e)","id": "stage_emissions_kg_co2e"},
        ],
        page_size=12, sort_action="native", filter_action="native",
        **table_styles(),
    )

    return html.Div([
        section_header("Supply Chain Stage Breakdown", [
            with_tooltip("Life cycle", "lca"), " emission fractions — ",
            with_tooltip("upstream", "upstream"), ", ",
            with_tooltip("operations", "operations"), ", ",
            with_tooltip("downstream", "downstream"), ", ",
            with_tooltip("transport", "transport"), ", and ",
            with_tooltip("end of life", "end_of_life"),
            " — across the product portfolio",
        ]),
        dbc.Row([
            dbc.Col(card_wrap(dcc.Graph(figure=fig_avg,   config={"displayModeBar": False})), xs=12, md=5),
            dbc.Col(card_wrap(dcc.Graph(figure=fig_stack, config={"displayModeBar": False})), xs=12, md=7),
        ], className="mb-3 g-3"),
        card_wrap(html.Div([
            html.Div(html.H6("Stage-Level Records", style={"color": TEXT_DARK, "margin": "0", "fontWeight": "600"}),
                     style={"backgroundColor": "#F0EDE8", "padding": "12px 16px", "borderBottom": f"1px solid {BORDER}"}),
            html.Div(tbl, style={"padding": "12px"}),
        ])),
    ])


def page_hotspots(company):
    df = filter_products(company)
    df_plot = df.dropna(subset=["upstream_frac","carbon_intensity"]).copy()
    df_plot["Classification"] = df_plot["is_hotspot"].map({True: "Hotspot", False: "Normal"})
    df_plot["product_short"]  = df_plot["product_name"].str[:45]

    fig_scatter = px.scatter(
        df_plot,
        x="upstream_frac", y="carbon_intensity",
        color="Classification",
        color_discrete_map={"Hotspot": HOTSPOT, "Normal": ACCENT},
        size="pcf_kg_co2e", size_max=28,
        hover_data={"product_short": True, "company": True, "pcf_kg_co2e": ":.2f"},
        labels={"upstream_frac": "Upstream Fraction", "carbon_intensity": "Carbon Intensity (kg CO2e/kg)"},
        title="K-Means Hotspot Detection -- Upstream Fraction vs Carbon Intensity",
    )
    chart_layout(fig_scatter, height=420)

    cluster_counts = df_plot.groupby("cluster_id").size().reset_index(name="count")
    cluster_counts["label"] = "Cluster " + cluster_counts["cluster_id"].astype(str)
    fig_pie = px.pie(
        cluster_counts, names="label", values="count",
        color_discrete_sequence=CHART_PALETTE,
        title="Product Distribution Across Clusters",
    )
    fig_pie.update_layout(paper_bgcolor=CARD_BG, font_color=TEXT_DARK,
                          title_font_size=13, height=300,
                          margin=dict(t=48, b=16, l=16, r=16))

    hotspots_df = df[df["is_hotspot"]].copy()
    hotspots_df["product_name"]     = hotspots_df["product_name"].str[:50]
    hotspots_df["pcf_kg_co2e"]      = hotspots_df["pcf_kg_co2e"].round(2)
    hotspots_df["carbon_intensity"] = hotspots_df["carbon_intensity"].round(2)
    hotspots_df["upstream_frac"]    = (hotspots_df["upstream_frac"] * 100).round(1)

    ts = table_styles(header_colour=HOTSPOT)
    tbl = dash_table.DataTable(
        data=hotspots_df[["product_name","company","pcf_kg_co2e","carbon_intensity","upstream_frac"]].to_dict("records"),
        columns=[
            {"name": "Product",           "id": "product_name"},
            {"name": "Company",           "id": "company"},
            {"name": "PCF (kg CO2e)",     "id": "pcf_kg_co2e"},
            {"name": "Carbon Intensity",  "id": "carbon_intensity"},
            {"name": "Upstream % of PCF", "id": "upstream_frac"},
        ],
        page_size=10, sort_action="native",
        **ts,
    )

    n_hotspots           = int(df["is_hotspot"].sum())
    hotspot_intensity    = df[df["is_hotspot"]]["carbon_intensity"].mean()
    nonhotspot_intensity = df[~df["is_hotspot"]]["carbon_intensity"].mean()

    return html.Div([
        section_header("ML Hotspot Detection", [
            with_tooltip("K-Means clustering", "kmeans"),
            " (k=4) on stage emission fractions. The cluster with highest combined ",
            with_tooltip("upstream", "upstream"),
            " and operations fraction is flagged as the ",
            with_tooltip("hotspot", "hotspot"),
            " cluster.",
        ]),
        dbc.Row([
            dbc.Col(kpi_card("Hotspot Products",       str(n_hotspots),                    "flagged products",          accent=HOTSPOT, tooltip_key="hotspot"),          xs=6, md=3),
            dbc.Col(kpi_card("Clusters",               "4",                                "K-Means k parameter",                       tooltip_key="kmeans"),           xs=6, md=3),
            dbc.Col(kpi_card("Hotspot Mean Intensity",  f"{hotspot_intensity:.2f} kg/kg",  "hotspot cluster",           accent=HOTSPOT, tooltip_key="carbon_intensity"), xs=6, md=3),
            dbc.Col(kpi_card("Normal Mean Intensity",   f"{nonhotspot_intensity:.2f} kg/kg","non-hotspot clusters",     accent=SUCCESS, tooltip_key="carbon_intensity"), xs=6, md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(card_wrap(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False})), xs=12, md=8),
            dbc.Col(card_wrap(dcc.Graph(figure=fig_pie,     config={"displayModeBar": False})), xs=12, md=4),
        ], className="mb-3 g-3"),
        card_wrap(html.Div([
            html.Div(html.H6("Flagged Hotspot Products", style={"color": HOTSPOT, "margin": "0", "fontWeight": "600"}),
                     style={"backgroundColor": "#FDF0EC", "padding": "12px 16px", "borderBottom": f"1px solid {BORDER}"}),
            html.Div(tbl, style={"padding": "12px"}),
        ])),
    ])


def page_benchmarking(custom_data=None):
    df = products.copy()
    company_stats = df.groupby("company").agg(
        mean_intensity=("carbon_intensity","mean"),
        total_pcf=("pcf_kg_co2e","sum"),
        n_products=("pcf_id","count"),
        mean_upstream=("upstream_frac","mean"),
    ).reset_index()
    company_stats = company_stats[company_stats["n_products"] >= 2].sort_values("mean_intensity")
    company_stats["mean_intensity"] = company_stats["mean_intensity"].round(2)
    company_stats["total_pcf"]      = company_stats["total_pcf"].round(1)
    company_stats["mean_upstream"]  = (company_stats["mean_upstream"] * 100).round(1)
    company_stats["mean_upstream"]  = company_stats["mean_upstream"].fillna("N/A")

    fig_bench = px.bar(
        company_stats, x="mean_intensity", y="company", orientation="h",
        color="mean_upstream", color_continuous_scale="YlOrBr",
        labels={"mean_intensity": "Mean Carbon Intensity (kg CO2e/kg)", "company": "", "mean_upstream": "Avg Upstream %"},
        title="Cross-Company Benchmarking — Mean Carbon Intensity",
        text="mean_intensity",
    )
    chart_layout(fig_bench, height=500)
    fig_bench.update_layout(yaxis={"categoryorder": "total ascending"},
                            coloraxis_colorbar=dict(title="Upstream %", tickfont=dict(color=TEXT_DARK)))
    fig_bench.update_traces(textposition="outside")

    if custom_data:
        for c in custom_data:
            fig_bench.add_trace(go.Bar(
                x=[c["mean_intensity"]],
                y=[f"★ {c['company']}"],
                orientation="h",
                marker_color=CUSTOM,
                marker_pattern_shape="/",
                showlegend=False,
                text=[f"{c['mean_intensity']:.2f}"],
                textposition="outside",
                hovertemplate=(
                    f"<b>★ {c['company']}</b><br>"
                    "Mean Intensity: %{x:.2f} kg CO₂e/kg<br>"
                    f"Upstream: {c['mean_upstream']:.0f}%<extra></extra>"
                ),
            ))
        fig_bench.update_layout(height=fig_bench.layout.height + 32 * len(custom_data))

    anchor = df[df["company"].isin(ANCHOR_COMPANIES)].groupby("company").agg(
        upstream=("upstream_frac","mean"),
        ops=("ops_frac","mean"),
        downstream=("downstream_frac","mean"),
        transport=("transport_frac","mean"),
    ).dropna()

    fig_radar = go.Figure()
    categories = ["Upstream","Operations","Downstream","Transport"]
    for i, (_, row) in enumerate(anchor.iterrows()):
        vals_pct = [row["upstream"]*100, row["ops"]*100, row["downstream"]*100, row["transport"]*100]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_pct + [vals_pct[0]],
            theta=categories + [categories[0]],
            fill="toself", opacity=0.45, name=row.name,
            line=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)]),
        ))

    if custom_data:
        for c in custom_data:
            vals = [c["upstream_frac"]*100, c["ops_frac"]*100,
                    c["downstream_frac"]*100, c["transport_frac"]*100]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself", opacity=0.55,
                name=f"★ {c['company']}",
                line=dict(color=CUSTOM, width=3, dash="dot"),
            ))

    fig_radar.update_layout(
        polar=dict(bgcolor=PAGE_BG,
                   radialaxis=dict(visible=True, color=TEXT_MID, gridcolor=BORDER),
                   angularaxis=dict(color=TEXT_MID)),
        paper_bgcolor=CARD_BG, font_color=TEXT_DARK,
        title="Anchor Company Stage Profile Comparison",
        title_font_size=13, height=420,
        margin=dict(t=48, b=16, l=40, r=40),
    )

    ts_bench = table_styles()
    display_stats = company_stats.copy()
    if custom_data:
        custom_rows = [{
            "company":        f"★ {c['company']}",
            "n_products":     1,
            "total_pcf":      round(c["total_pcf"], 1),
            "mean_intensity": c["mean_intensity"],
            "mean_upstream":  c["mean_upstream"],
        } for c in custom_data]
        display_stats = pd.concat(
            [pd.DataFrame(custom_rows), display_stats], ignore_index=True
        )
        ts_bench["style_data_conditional"].append({
            "if": {"filter_query": '{company} contains "★"'},
            "backgroundColor": "#EBF8F8",
            "fontWeight": "700",
            "color": CUSTOM,
        })

    return html.Div([
        section_header("Cross-Company Benchmarking", [
            "Mean ", with_tooltip("carbon intensity", "carbon_intensity"),
            " and stage profile comparison across the F&B dataset",
        ]),
        dbc.Row([
            dbc.Col(card_wrap(dcc.Graph(figure=fig_bench, config={"displayModeBar": False})), xs=12, md=7),
            dbc.Col(card_wrap(dcc.Graph(figure=fig_radar, config={"displayModeBar": False})), xs=12, md=5),
        ], className="mb-3 g-3"),
        card_wrap(html.Div([
            html.Div(html.H6("Company Summary Statistics", style={"color": TEXT_DARK, "margin": "0", "fontWeight": "600"}),
                     style={"backgroundColor": "#F0EDE8", "padding": "12px 16px", "borderBottom": f"1px solid {BORDER}"}),
            html.Div(dash_table.DataTable(
                data=display_stats.to_dict("records"),
                columns=[
                    {"name": "Company",                "id": "company"},
                    {"name": "Products",               "id": "n_products"},
                    {"name": "Total PCF (kg CO2e)",    "id": "total_pcf"},
                    {"name": "Mean Intensity (kg/kg)", "id": "mean_intensity"},
                    {"name": "Avg Upstream %",         "id": "mean_upstream", "type": "text"},
                ],
                sort_action="native",
                **ts_bench,
            ), style={"padding": "12px"}),
        ])),
    ])


def page_interventions():
    df = INTERVENTIONS.copy()

    fig_roi = px.bar(
        df.sort_values("roi_score", ascending=True),
        x="roi_score", y="description", orientation="h",
        color="stage", color_discrete_map=STAGE_COLOURS,
        labels={"roi_score": "ROI Score (0-1)", "description": ""},
        title="Intervention Ranking by ROI Score",
        text="roi_score",
    )
    chart_layout(fig_roi, height=400)
    fig_roi.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_roi.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    fig_bubble = px.scatter(
        df, x="est_cost_gbp", y="est_co2e_saving_tonne",
        size="roi_score", color="stage", color_discrete_map=STAGE_COLOURS,
        hover_data={"description": True, "payback_years": True},
        labels={"est_cost_gbp": "Estimated Cost (GBP)", "est_co2e_saving_tonne": "CO2e Saving (tonnes)"},
        title="Cost vs Emission Saving  (bubble size = ROI score)",
    )
    chart_layout(fig_bubble, height=380)

    df_display = df.copy()
    df_display["roi_score"]             = df_display["roi_score"].round(2)
    df_display["est_co2e_saving_tonne"] = df_display["est_co2e_saving_tonne"].round(1)
    df_display["est_cost_gbp"]          = df_display["est_cost_gbp"].apply(lambda x: f"GBP {x:,.0f}")

    ts = table_styles()
    ts["style_data_conditional"] = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"},
        {"if": {"filter_query": "{roi_score} > 0.85"}, "color": SUCCESS, "fontWeight": "600"},
    ]
    tbl = dash_table.DataTable(
        data=df_display[["stage","description","est_co2e_saving_tonne","est_cost_gbp","roi_score","payback_years"]].to_dict("records"),
        columns=[
            {"name": "Stage",           "id": "stage"},
            {"name": "Intervention",    "id": "description"},
            {"name": "CO2e Saving (t)", "id": "est_co2e_saving_tonne"},
            {"name": "Est. Cost",       "id": "est_cost_gbp"},
            {"name": "ROI Score",       "id": "roi_score"},
            {"name": "Payback (yrs)",   "id": "payback_years"},
        ],
        sort_action="native",
        **ts,
    )

    return html.Div([
        section_header("Cost-Benefit Intervention Ranking", [
            "Interventions scored by estimated CO₂e reduction per GBP spent. ",
            with_tooltip("ROI Score", "roi_score"), " higher = better return. ",
            with_tooltip("Payback period", "payback"),
            " calculated at £50 / tonne CO₂e. GreenMart scenario (synthetic data).",
        ]),
        dbc.Row([
            dbc.Col(card_wrap(dcc.Graph(figure=fig_roi,    config={"displayModeBar": False})), xs=12, md=6),
            dbc.Col(card_wrap(dcc.Graph(figure=fig_bubble, config={"displayModeBar": False})), xs=12, md=6),
        ], className="mb-3 g-3"),
        card_wrap(html.Div([
            html.Div(html.H6("Ranked Intervention Table  --  bold green = ROI above 0.85", style={"color": TEXT_DARK, "margin": "0", "fontWeight": "600"}),
                     style={"backgroundColor": "#F0EDE8", "padding": "12px 16px", "borderBottom": f"1px solid {BORDER}"}),
            html.Div(tbl, style={"padding": "12px"}),
        ])),
    ])


# ── Glossary toggle ───────────────────────────────────────────────────────────
@callback(
    Output("glossary-offcanvas", "is_open"),
    Input("open-glossary-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_glossary(_):
    return True


@callback(
    Output("glossary-offcanvas", "is_open", allow_duplicate=True),
    Input("open-glossary-btn-mobile", "n_clicks"),
    prevent_initial_call=True,
)
def open_glossary_mobile(_):
    return True


# ── Mobile sidebar toggle ─────────────────────────────────────────────────────
@callback(
    Output("mobile-sidebar-offcanvas", "is_open"),
    Input("mobile-menu-btn", "n_clicks"),
    State("mobile-sidebar-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_mobile_sidebar(_, is_open):
    return not is_open


# ── Sync mobile company filter → desktop (drives all existing callbacks) ──────
@callback(
    Output("company-filter", "value", allow_duplicate=True),
    Input("company-filter-mobile", "value"),
    prevent_initial_call=True,
)
def sync_company_from_mobile(val):
    return val or "All Companies"


# ── Router ────────────────────────────────────────────────────────────────────
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("company-filter", "value"),
    Input("custom-company-store", "data"),
)
def render_page(pathname, company, custom_data):
    if pathname in ["/", None]:
        return page_overview(company)
    elif pathname == "/stages":
        return page_stages(company)
    elif pathname == "/hotspots":
        return page_hotspots(company)
    elif pathname == "/benchmarking":
        return page_benchmarking(custom_data)
    elif pathname == "/interventions":
        return page_interventions()
    return html.H4("Page not found", style={"color": TEXT_DARK})


# ── New-company modal open/close ──────────────────────────────────────────────
@callback(
    Output("new-company-modal", "is_open"),
    Input("company-filter", "value"),
    Input("nc-cancel", "n_clicks"),
    Input("editing-company-store", "data"),
    prevent_initial_call=True,
)
def toggle_nc_modal(company_val, _cancel, editing_company):
    if ctx.triggered_id == "company-filter":
        return company_val == "__new__"
    if ctx.triggered_id == "editing-company-store" and editing_company:
        return True
    return False


# ── Live stage-percentage sum ─────────────────────────────────────────────────
@callback(
    Output("nc-stage-sum", "children"),
    Output("nc-stage-sum", "style"),
    Input("nc-upstream", "value"),
    Input("nc-ops", "value"),
    Input("nc-downstream", "value"),
    Input("nc-transport", "value"),
    Input("nc-eol", "value"),
)
def update_stage_sum(up, ops, dn, tr, eol):
    total = sum(v or 0 for v in [up, ops, dn, tr, eol])
    if total == 0:
        colour = TEXT_MID
    elif abs(total - 100) <= 1:
        colour = SUCCESS
    else:
        colour = HOTSPOT
    return f"Total: {total:.0f} %", {
        "fontSize": "0.82rem", "fontWeight": "700",
        "color": colour, "lineHeight": "32px",
    }


# ── Store write (submit + clear) ──────────────────────────────────────────────
@callback(
    Output("custom-company-store", "data"),
    Output("nc-error", "children"),
    Output("company-filter", "value"),
    Output("url", "pathname"),
    Output("editing-company-store", "data", allow_duplicate=True),
    Input("nc-submit", "n_clicks"),
    State("nc-name", "value"),
    State("nc-product", "value"),
    State("nc-pcf", "value"),
    State("nc-weight", "value"),
    State("nc-upstream", "value"),
    State("nc-ops", "value"),
    State("nc-downstream", "value"),
    State("nc-transport", "value"),
    State("nc-eol", "value"),
    State("custom-company-store", "data"),
    State("url", "pathname"),
    State("editing-company-store", "data"),
    prevent_initial_call=True,
)
def handle_store(submit_n,
                 name, product, pcf, weight, up, ops, dn, tr, eol,
                 existing_data, current_path, editing_company):
    if not (name and product and pcf is not None and weight is not None):
        return no_update, "Please fill in all required fields.", no_update, no_update, no_update

    if float(weight) <= 0:
        return no_update, "Product weight must be greater than zero.", no_update, no_update, no_update

    total_pct = sum(v or 0 for v in [up, ops, dn, tr, eol])
    if abs(total_pct - 100) > 1:
        return (no_update,
                f"Stage percentages sum to {total_pct:.1f} %. They must sum to 100 %.",
                no_update, no_update, no_update)

    pcf_f, wt_f = float(pcf), float(weight)
    up_f, ops_f, dn_f, tr_f, eol_f = [float(v or 0) for v in [up, ops, dn, tr, eol]]

    record = {
        "company":        name.strip(),
        "product_name":   product.strip(),
        "pcf_kg_co2e":    round(pcf_f, 4),
        "carbon_intensity": round(pcf_f / wt_f, 4),
        "upstream_frac":  up_f  / 100,
        "ops_frac":       ops_f / 100,
        "downstream_frac": dn_f / 100,
        "transport_frac": tr_f  / 100,
        "endoflife_frac": eol_f / 100,
        "mean_intensity": round(pcf_f / wt_f, 2),
        "mean_upstream":  up_f,
        "total_pcf":      round(pcf_f, 2),
    }
    # Remove old entry by new name AND by original name if editing with a renamed company
    data = [r for r in (existing_data or [])
            if r["company"] != record["company"] and r["company"] != (editing_company or "")]
    data.append(record)
    return data, "", "All Companies", "/benchmarking", None


# ── Sidebar active-company banner ─────────────────────────────────────────────
@callback(
    Output("nc-active-banner", "style"),
    Output("nc-active-banner", "children"),
    Input("custom-company-store", "data"),
)
def show_nc_banner(data):
    if not data:
        return {"display": "none"}, []
    btn_style = {
        "background": "none", "border": "none", "color": "#8FA8BF",
        "cursor": "pointer", "padding": "0 3px", "fontSize": "0.75rem", "flexShrink": "0",
    }
    rows = []
    for r in data:
        rows.append(html.Div([
            html.Span(f"📊 {r['company']}", style={
                "color": "#A8DEAD", "fontSize": "0.72rem", "fontWeight": "600",
                "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
                "flex": "1",
            }),
            html.Button("✏", id={"type": "nc-edit-btn", "index": r["company"]},
                        n_clicks=0, title="Edit", style=btn_style),
            html.Button("✕", id={"type": "nc-delete-btn", "index": r["company"]},
                        n_clicks=0, title="Remove", style=btn_style),
        ], style={
            "display": "flex", "alignItems": "center",
            "backgroundColor": "#1A3344", "borderRadius": "4px",
            "padding": "5px 8px", "borderLeft": f"3px solid {CUSTOM}",
            "marginBottom": "4px",
        }))
    return {"display": "block", "marginBottom": "16px"}, rows


@callback(
    Output("custom-company-store", "data", allow_duplicate=True),
    Input({"type": "nc-delete-btn", "index": ALL}, "n_clicks"),
    State("custom-company-store", "data"),
    prevent_initial_call=True,
)
def delete_company(n_clicks_list, data):
    if not any(n or 0 for n in n_clicks_list) or not data:
        return no_update
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    return [r for r in data if r["company"] != triggered["index"]]


@callback(
    Output("editing-company-store", "data"),
    Input({"type": "nc-edit-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_editing_company(n_clicks_list):
    if not any(n or 0 for n in n_clicks_list):
        return no_update
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    return triggered["index"]


@callback(
    Output("nc-name",       "value"),
    Output("nc-product",    "value"),
    Output("nc-pcf",        "value"),
    Output("nc-weight",     "value"),
    Output("nc-upstream",   "value"),
    Output("nc-ops",        "value"),
    Output("nc-downstream", "value"),
    Output("nc-transport",  "value"),
    Output("nc-eol",        "value"),
    Input("editing-company-store", "data"),
    State("custom-company-store", "data"),
    prevent_initial_call=True,
)
def populate_edit_form(editing_company, all_data):
    if not editing_company or not all_data:
        return (None,) * 9
    record = next((r for r in all_data if r["company"] == editing_company), None)
    if not record:
        return (None,) * 9
    weight = round(record["pcf_kg_co2e"] / record["carbon_intensity"], 3) if record.get("carbon_intensity") else None
    return (
        record["company"],
        record["product_name"],
        record["pcf_kg_co2e"],
        weight,
        round(record["upstream_frac"] * 100, 1),
        round(record["ops_frac"] * 100, 1),
        round(record["downstream_frac"] * 100, 1),
        round(record["transport_frac"] * 100, 1),
        round(record["endoflife_frac"] * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
