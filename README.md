# GreenMart Carbon Analytics Platform

An enterprise carbon footprint analytics platform for retail supply chain emissions, built as a final-year dissertation project at De Montfort University.

The platform uses the [Carbon Catalogue dataset](https://doi.org/10.6084/m9.figshare.c.5408100) (Kaack et al., 2022) to demonstrate GHG Protocol-compliant Scope 1/2/3 emissions analysis for a fictional retailer, GreenMart, modelled on five anchor companies from the Food, Beverages and Tobacco sector.

---

## Features

- **Emissions calculation engine** implementing GHG Protocol Scope 1, 2, and 3 methodologies
- **K-Means clustering** for emissions hotspot detection across product portfolios
- **Cost-benefit intervention ranking** using a custom ROI score (estimated CO2e reduction per unit cost)
- **Interactive ESG dashboard** built with Plotly Dash, including stage profile comparisons, carbon intensity distributions, and benchmark visualisations
- **Anchor company analysis** across five companies: Coca-Cola Enterprises, Barilla Holding SpA, Nestlé, Danone, and CJ Cheiljedang

---

## Project Structure

```
greenmart-carbon/
├── app.py                  # Dash application entry point
├── clean_data.py           # Data cleaning and preprocessing pipeline
├── assets/                 # Stylesheets and static files
├── data/                   # Cleaned output data (raw source files excluded)
│   ├── fb_products_clean.csv
│   └── fb_stages_clean.csv
├── notebook/
│   └── carbon-catalogue.ipynb   # EDA notebook
├── pages/                  # Individual Dash page modules
└── requirements.txt
```

---

## Dataset

This project uses the Carbon Catalogue (Kaack et al., 2022), an open-access dataset of 866 verified product carbon footprints from 145 companies across 8 sectors.

Download the raw data from Figshare:
**[https://doi.org/10.6084/m9.figshare.c.5408100](https://doi.org/10.6084/m9.figshare.c.5408100)**

Place the downloaded Excel file in the project root, then run the cleaning pipeline:

```bash
python clean_data.py
```

This produces `fb_products_clean.csv` (139 rows) and `fb_stages_clean.csv` (351 rows) in the `data/` folder. The platform is scoped to the Food, Beverages and Tobacco sector.

---

## Setup

**Requirements:** Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
python app.py
```

Then open your browser and go to `http://127.0.0.1:8050`.

---

## Methodology Notes

- Stage-level emissions are derived by multiplying total PCF by the proportional stage fractions provided in the Carbon Catalogue. They are not absolute values in the source data.
- Carbon intensity is calculated as total PCF divided by product weight (kg CO2e per kg of product), enabling fair cross-product comparison.
- The ROI score is a project-designed cost-effectiveness ratio: estimated CO2e reduction divided by estimated intervention cost. It is not sourced from an external reference.
- K-Means clustering is applied to carbon intensity and stage-level features to identify emissions hotspots. Cluster labels are interpreted in business terms post-hoc.

---

## References

Kaack, L.H., Apt, J., Morgan, M.G. and Whitacre, J.F. (2022). The Carbon Catalogue: Carbon footprints of 866 commercial products from 8 industry sectors and 5 continents. *Scientific Data*, 9(1), p.87. DOI: [10.1038/s41597-022-01178-9](https://doi.org/10.1038/s41597-022-01178-9)

---

## Author

Chidiere Oluoma - BSc Computer Science, De Montfort University  
Supervised by Dr. Lipika Deka