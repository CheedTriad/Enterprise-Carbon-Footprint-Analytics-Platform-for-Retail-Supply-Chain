"""
clean_data.py
Reads the Carbon Catalogue Excel file, filters to Food & Beverage sector,
cleans column names, and writes two CSV files:
  - fb_products_clean.csv  (product-level, one row per product)
  - fb_stages_clean.csv    (stage-level, multiple rows per product)
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

EXCEL_PATH = r"C:\Users\chidi\Documents\greenmart-carbon\PublicTablesForCarbonCatalogueDataDescriptor_v30Oct2021.xlsx"

# ── 1. Load raw sheets ─────────────────────────────────────────────────────────
products_raw = pd.read_excel(EXCEL_PATH, sheet_name="Product Level Data")
stages_raw   = pd.read_excel(EXCEL_PATH, sheet_name="Stage Level Data")

# ── 2. Rename columns to clean snake_case ─────────────────────────────────────
product_rename = {
    "*PCF-ID":                                  "pcf_id",
    "Year of reporting":                        "year",
    "*Stage-level CO2e available":              "has_stages",
    "Product name (and functional unit)":       "product_name",
    "Product detail":                           "product_detail",
    "Company":                                  "company",
    "Country (where company is incorporated)":  "country",
    "Company's GICS Industry Group":            "gics_group",
    "Company's GICS Industry":                  "gics_industry",
    "*Company's sector":                        "sector",
    "Product weight (kg)":                      "weight_kg",
    "*Source for product weight":               "weight_source",
    "Product's carbon footprint (PCF, kg CO2e)":"pcf_kg_co2e",
    "*Carbon intensity":                        "carbon_intensity",
    "Protocol used for PCF":                    "protocol",
    "Relative change in PCF vs previous":       "pct_change_vs_prev",
    "Company-reported reason for change":       "change_reason_reported",
    "*Change reason category":                  "change_reason_category",
    "*%Upstream estimated from %Operations":    "upstream_estimated",
    "*Upstream CO2e (fraction of total PCF)":   "upstream_frac",
    "*Operations CO2e (fraction of total PCF)": "ops_frac",
    "*Downstream CO2e (fraction of total PCF)": "downstream_frac",
    "*Transport CO2e (fraction of total PCF)":  "transport_frac",
    "*EndOfLife CO2e (fraction of total PCF)":  "endoflife_frac",
    "*Adjustments to raw data (if any)":        "adjustments",
}

stage_rename = {
    "*PCF-ID":                                              "pcf_id",
    "Description of LCA stage":                            "stage_desc",
    "Scope-characterization of LCA stage":                 "scope",
    "*Assigned value chain portion":                        "value_chain",
    "Emissions at stage (kg CO2e)":                        "stage_emissions_kg_co2e",
    "*Emissions at this stage are exclusively transport":   "is_transport",
    "*Emissions at this stage are exclusively EndOfLife":   "is_endoflife",
}

products_raw.rename(columns=product_rename, inplace=True)
stages_raw.rename(columns=stage_rename, inplace=True)

# ── 3. Filter to Food & Beverage ──────────────────────────────────────────────
fb_products = products_raw[products_raw["sector"] == "Food & Beverage"].copy()
fb_ids      = set(fb_products["pcf_id"])
fb_stages   = stages_raw[stages_raw["pcf_id"].isin(fb_ids)].copy()

# ── 4. Type coercion ──────────────────────────────────────────────────────────
fb_products["pcf_kg_co2e"]      = pd.to_numeric(fb_products["pcf_kg_co2e"],      errors="coerce")
fb_products["carbon_intensity"] = pd.to_numeric(fb_products["carbon_intensity"],  errors="coerce")
fb_products["weight_kg"]        = pd.to_numeric(fb_products["weight_kg"],         errors="coerce")
fb_products["year"]             = pd.to_numeric(fb_products["year"],              errors="coerce").astype("Int64")

for frac_col in ["upstream_frac","ops_frac","downstream_frac","transport_frac","endoflife_frac"]:
    fb_products[frac_col] = pd.to_numeric(fb_products[frac_col], errors="coerce")

fb_stages["stage_emissions_kg_co2e"] = pd.to_numeric(fb_stages["stage_emissions_kg_co2e"], errors="coerce")

# ── 5. Drop rows missing the core metric ──────────────────────────────────────
fb_products.dropna(subset=["pcf_kg_co2e"], inplace=True)
fb_stages.dropna(subset=["stage_emissions_kg_co2e"], inplace=True)

# ── 6. Derived columns ────────────────────────────────────────────────────────
# Absolute stage emissions per fraction (used for dashboard charts)
for frac_col, abs_col in [
    ("upstream_frac",   "upstream_co2e"),
    ("ops_frac",        "ops_co2e"),
    ("downstream_frac", "downstream_co2e"),
    ("transport_frac",  "transport_co2e"),
    ("endoflife_frac",  "endoflife_co2e"),
]:
    fb_products[abs_col] = fb_products["pcf_kg_co2e"] * fb_products[frac_col]

# ── 7. Write output CSVs ──────────────────────────────────────────────────────
fb_products.to_csv("fb_products_clean.csv", index=False)
fb_stages.to_csv("fb_stages_clean.csv",   index=False)

print(f"fb_products_clean.csv  →  {len(fb_products)} rows, {fb_products.shape[1]} columns")
print(f"fb_stages_clean.csv    →  {len(fb_stages)} rows, {fb_stages.shape[1]} columns")
print(f"\nCompanies in dataset:")
print(fb_products["company"].value_counts().to_string())
print(f"\nNull counts in key columns:")
print(fb_products[["pcf_kg_co2e","carbon_intensity","upstream_frac","ops_frac","downstream_frac"]].isnull().sum())
