"""
Capital Gains Tax Sensitivity Analysis for California Residents (2026)

This script analyzes the tax implications of selling private company shares
under both Long-Term Capital Gains (LTCG) and Short-Term Capital Gains (STCG)
scenarios, varying ordinary income and sale price per share.

Usage:
    1. Copy config.example.json to config.json
    2. Edit config.json with your personal parameters
    3. Run: python capital_gains_tax_analysis.py

Author: Generated for tax planning purposes

DISCLAIMER:
    This tool is for educational and planning purposes only. Tax laws are
    complex and subject to change. Always consult a qualified tax professional.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

NOTE ON EXAMPLE DATA:
    The values in config.example.json are placeholder representations only,
    using round numbers (e.g., $1.00, 10,000 shares) for illustration purposes.
    They do not represent any actual person's financial situation.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from JSON file.

    Priority:
    1. Explicit path passed as argument
    2. config.json in current directory (private, gitignored)
    3. config.example.json (public example with placeholder values)

    Returns normalized config dict with consistent structure.
    """
    script_dir = Path(__file__).parent

    # Determine which config file to use
    if config_path:
        config_file = Path(config_path)
    elif (script_dir / "config.json").exists():
        config_file = script_dir / "config.json"
    elif (script_dir / "config.example.json").exists():
        config_file = script_dir / "config.example.json"
        print("WARNING: Using config.example.json - copy to config.json and customize")
    else:
        raise FileNotFoundError(
            "No configuration file found. Please create config.json from config.example.json"
        )

    print(f"Loading configuration from: {config_file}")

    with open(config_file, "r") as f:
        raw_config = json.load(f)

    # Normalize config structure for internal use
    config = {
        "cost_basis_per_share": raw_config["cost_basis_per_share"],
        "num_shares": raw_config["num_shares"],
        "filing_status": raw_config["filing_status"],
        "tax_year": raw_config.get("tax_year", 2026),
        "income_range": (
            raw_config["income_range"]["min"],
            raw_config["income_range"]["max"]
        ),
        "income_steps": raw_config["income_range"]["steps"],
        "price_range": (
            raw_config["price_range"]["min"],
            raw_config["price_range"]["max"]
        ),
        "price_steps": raw_config["price_range"]["steps"],
        "representative_scenarios": raw_config.get("representative_scenarios", [
            {"income": 150000, "price": 20},
            {"income": 250000, "price": 30},
            {"income": 400000, "price": 40},
        ]),
    }

    return config

# =============================================================================
# 2026 TAX BRACKETS (Estimated - based on inflation adjustments from 2024)
# These are estimates; actual 2026 brackets may differ slightly
# =============================================================================

# Federal Standard Deduction (2026 estimates)
STANDARD_DEDUCTION = {
    "single": 15_000,
    "married": 30_000,
}

# Federal Ordinary Income Tax Brackets 2026 (Single)
# Format: (upper_limit, rate)
FEDERAL_ORDINARY_BRACKETS_SINGLE = [
    (11_600, 0.10),
    (47_150, 0.12),
    (100_525, 0.22),
    (191_950, 0.24),
    (243_725, 0.32),
    (609_350, 0.35),
    (float('inf'), 0.37),
]

# Federal Ordinary Income Tax Brackets 2026 (Married Filing Jointly)
FEDERAL_ORDINARY_BRACKETS_MARRIED = [
    (23_200, 0.10),
    (94_300, 0.12),
    (201_050, 0.22),
    (383_900, 0.24),
    (487_450, 0.32),
    (731_200, 0.35),
    (float('inf'), 0.37),
]

# Federal Long-Term Capital Gains Brackets 2026 (Single)
# Based on taxable income levels
FEDERAL_LTCG_BRACKETS_SINGLE = [
    (47_025, 0.00),    # 0% rate up to this taxable income
    (518_900, 0.15),   # 15% rate up to this taxable income
    (float('inf'), 0.20),  # 20% rate above
]

# Federal Long-Term Capital Gains Brackets 2026 (Married Filing Jointly)
FEDERAL_LTCG_BRACKETS_MARRIED = [
    (94_050, 0.00),
    (583_750, 0.15),
    (float('inf'), 0.20),
]

# California State Tax Brackets 2026 (Single) - CA taxes capital gains as ordinary income
# Format: (upper_limit, rate)
CA_BRACKETS_SINGLE = [
    (10_412, 0.01),
    (24_684, 0.02),
    (38_959, 0.04),
    (54_081, 0.06),
    (68_350, 0.08),
    (349_137, 0.093),
    (418_961, 0.103),
    (698_271, 0.113),
    (float('inf'), 0.123),
]

# Mental Health Services Tax (additional 1% on income over $1M)
CA_MENTAL_HEALTH_THRESHOLD = 1_000_000
CA_MENTAL_HEALTH_RATE = 0.01

# California State Tax Brackets 2026 (Married Filing Jointly)
CA_BRACKETS_MARRIED = [
    (20_824, 0.01),
    (49_368, 0.02),
    (77_918, 0.04),
    (108_162, 0.06),
    (136_700, 0.08),
    (698_274, 0.093),
    (837_922, 0.103),
    (1_396_542, 0.113),
    (float('inf'), 0.123),
]

# Net Investment Income Tax (NIIT)
# 3.8% on net investment income when MAGI exceeds threshold
NIIT_RATE = 0.038
NIIT_THRESHOLD = {
    "single": 200_000,
    "married": 250_000,
}


# =============================================================================
# TAX CALCULATION FUNCTIONS
# =============================================================================

def calculate_bracket_tax(income: float, brackets: list) -> float:
    """
    Calculate tax using progressive brackets.

    Args:
        income: Taxable income
        brackets: List of (upper_limit, rate) tuples

    Returns:
        Total tax amount
    """
    if income <= 0:
        return 0

    tax = 0
    prev_limit = 0

    for upper_limit, rate in brackets:
        if income <= prev_limit:
            break
        taxable_in_bracket = min(income, upper_limit) - prev_limit
        tax += taxable_in_bracket * rate
        prev_limit = upper_limit

    return tax


def calculate_federal_ordinary_tax(taxable_income: float, filing_status: str) -> float:
    """Calculate federal ordinary income tax."""
    brackets = (FEDERAL_ORDINARY_BRACKETS_SINGLE if filing_status == "single"
                else FEDERAL_ORDINARY_BRACKETS_MARRIED)
    return calculate_bracket_tax(taxable_income, brackets)


def calculate_federal_ltcg_tax(ordinary_taxable_income: float,
                                capital_gain: float,
                                filing_status: str) -> float:
    """
    Calculate federal long-term capital gains tax.

    LTCG is taxed based on where total taxable income falls in the brackets,
    but the gain itself is taxed at preferential rates.
    """
    brackets = (FEDERAL_LTCG_BRACKETS_SINGLE if filing_status == "single"
                else FEDERAL_LTCG_BRACKETS_MARRIED)

    total_taxable = ordinary_taxable_income + capital_gain

    # Calculate tax on ordinary income portion at LTCG rates (for stacking)
    # Then calculate tax on total, subtract the ordinary portion

    tax = 0
    prev_limit = 0
    remaining_gain = capital_gain
    current_income = ordinary_taxable_income

    for upper_limit, rate in brackets:
        if remaining_gain <= 0:
            break

        # How much room is there in this bracket above our ordinary income?
        if current_income >= upper_limit:
            prev_limit = upper_limit
            continue

        bracket_room = upper_limit - max(current_income, prev_limit)
        gain_in_bracket = min(remaining_gain, bracket_room)

        tax += gain_in_bracket * rate
        remaining_gain -= gain_in_bracket
        current_income += gain_in_bracket
        prev_limit = upper_limit

    return tax


def calculate_ca_tax(taxable_income: float, filing_status: str) -> float:
    """
    Calculate California state income tax.
    CA taxes capital gains as ordinary income.
    Includes Mental Health Services Tax (1% on income > $1M).
    """
    brackets = (CA_BRACKETS_SINGLE if filing_status == "single"
                else CA_BRACKETS_MARRIED)

    tax = calculate_bracket_tax(taxable_income, brackets)

    # Mental Health Services Tax
    if taxable_income > CA_MENTAL_HEALTH_THRESHOLD:
        tax += (taxable_income - CA_MENTAL_HEALTH_THRESHOLD) * CA_MENTAL_HEALTH_RATE

    return tax


def calculate_niit(magi: float, investment_income: float, filing_status: str) -> float:
    """
    Calculate Net Investment Income Tax (NIIT).

    NIIT is 3.8% on the LESSER of:
    - Net investment income, OR
    - MAGI exceeding the threshold
    """
    threshold = NIIT_THRESHOLD[filing_status]

    if magi <= threshold:
        return 0

    excess_magi = magi - threshold
    taxable_amount = min(investment_income, excess_magi)

    return taxable_amount * NIIT_RATE


def calculate_total_tax_ltcg(ordinary_income: float,
                              capital_gain: float,
                              filing_status: str) -> dict:
    """
    Calculate total tax liability for LONG-TERM capital gains scenario.

    Returns detailed breakdown of all tax components.
    """
    std_deduction = STANDARD_DEDUCTION[filing_status]

    # Federal calculations
    ordinary_taxable = max(0, ordinary_income - std_deduction)

    # Federal tax on ordinary income
    federal_ordinary_tax = calculate_federal_ordinary_tax(ordinary_taxable, filing_status)

    # Federal LTCG tax (preferential rates)
    federal_ltcg_tax = calculate_federal_ltcg_tax(ordinary_taxable, capital_gain, filing_status)

    # California tax (taxes all income including LTCG as ordinary)
    # CA has its own standard deduction but for simplicity using federal AGI concept
    ca_taxable = ordinary_income + capital_gain  # CA doesn't have standard deduction like federal
    # Actually CA has a much smaller standard deduction, let's account for it
    ca_std_deduction = 5_363 if filing_status == "single" else 10_726  # 2024 values, approx for 2026
    ca_taxable = max(0, ca_taxable - ca_std_deduction)
    ca_tax = calculate_ca_tax(ca_taxable, filing_status)

    # NIIT - applies to investment income when MAGI exceeds threshold
    magi = ordinary_income + capital_gain  # Simplified MAGI
    niit = calculate_niit(magi, capital_gain, filing_status)

    total_federal = federal_ordinary_tax + federal_ltcg_tax + niit
    total_tax = total_federal + ca_tax

    return {
        "ordinary_income": ordinary_income,
        "capital_gain": capital_gain,
        "federal_ordinary_tax": federal_ordinary_tax,
        "federal_ltcg_tax": federal_ltcg_tax,
        "niit": niit,
        "ca_tax": ca_tax,
        "total_federal": total_federal,
        "total_tax": total_tax,
        "effective_rate_on_gain": (federal_ltcg_tax + niit +
                                   (ca_tax - calculate_ca_tax(max(0, ordinary_income - ca_std_deduction), filing_status))) / capital_gain if capital_gain > 0 else 0,
    }


def calculate_total_tax_stcg(ordinary_income: float,
                              capital_gain: float,
                              filing_status: str) -> dict:
    """
    Calculate total tax liability for SHORT-TERM capital gains scenario.

    STCG is taxed as ordinary income at federal level.
    """
    std_deduction = STANDARD_DEDUCTION[filing_status]

    # Federal calculations - STCG added to ordinary income
    total_ordinary = ordinary_income + capital_gain
    taxable_income = max(0, total_ordinary - std_deduction)

    # Calculate tax on just ordinary income for comparison
    ordinary_taxable = max(0, ordinary_income - std_deduction)
    federal_tax_ordinary_only = calculate_federal_ordinary_tax(ordinary_taxable, filing_status)

    # Total federal tax (STCG taxed as ordinary income)
    federal_tax_total = calculate_federal_ordinary_tax(taxable_income, filing_status)
    federal_stcg_tax = federal_tax_total - federal_tax_ordinary_only

    # California tax (same treatment as LTCG - ordinary income rates)
    ca_std_deduction = 5_363 if filing_status == "single" else 10_726
    ca_taxable = max(0, total_ordinary - ca_std_deduction)
    ca_tax = calculate_ca_tax(ca_taxable, filing_status)

    # NIIT still applies to investment income
    magi = total_ordinary
    niit = calculate_niit(magi, capital_gain, filing_status)

    total_federal = federal_tax_total + niit
    total_tax = total_federal + ca_tax

    return {
        "ordinary_income": ordinary_income,
        "capital_gain": capital_gain,
        "federal_ordinary_tax": federal_tax_ordinary_only,
        "federal_stcg_tax": federal_stcg_tax,
        "niit": niit,
        "ca_tax": ca_tax,
        "total_federal": total_federal,
        "total_tax": total_tax,
        "effective_rate_on_gain": (federal_stcg_tax + niit +
                                   (ca_tax - calculate_ca_tax(max(0, ordinary_income - ca_std_deduction), filing_status))) / capital_gain if capital_gain > 0 else 0,
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_sensitivity_analysis(config: dict) -> dict:
    """
    Run full two-parameter sensitivity analysis.

    Returns matrices for LTCG and STCG total taxes across income/price grid,
    including breakdowns by federal, state, and NIIT components.
    """
    # Create parameter grids
    incomes = np.linspace(config["income_range"][0],
                          config["income_range"][1],
                          config["income_steps"])

    prices = np.linspace(config["price_range"][0],
                         config["price_range"][1],
                         config["price_steps"])

    # Initialize result matrices
    n_income = len(incomes)
    n_price = len(prices)

    # Total taxes
    ltcg_total_tax = np.zeros((n_price, n_income))
    stcg_total_tax = np.zeros((n_price, n_income))

    # Federal components (capital gains portion only)
    ltcg_federal_tax = np.zeros((n_price, n_income))
    stcg_federal_tax = np.zeros((n_price, n_income))

    # State (CA) tax
    ltcg_ca_tax = np.zeros((n_price, n_income))
    stcg_ca_tax = np.zeros((n_price, n_income))

    # NIIT
    ltcg_niit = np.zeros((n_price, n_income))
    stcg_niit = np.zeros((n_price, n_income))

    # Rates and other
    ltcg_effective_rate = np.zeros((n_price, n_income))
    stcg_effective_rate = np.zeros((n_price, n_income))
    capital_gains = np.zeros((n_price, n_income))
    niit_applies = np.zeros((n_price, n_income))

    filing_status = config["filing_status"]

    for i, price in enumerate(prices):
        # Calculate capital gain for this price
        gain_per_share = price - config["cost_basis_per_share"]
        total_gain = gain_per_share * config["num_shares"]

        for j, income in enumerate(incomes):
            capital_gains[i, j] = total_gain

            # LTCG scenario
            ltcg_result = calculate_total_tax_ltcg(income, total_gain, filing_status)
            ltcg_total_tax[i, j] = ltcg_result["total_tax"]
            ltcg_federal_tax[i, j] = ltcg_result["federal_ltcg_tax"]
            ltcg_ca_tax[i, j] = ltcg_result["ca_tax"]
            ltcg_niit[i, j] = ltcg_result["niit"]
            ltcg_effective_rate[i, j] = ltcg_result["effective_rate_on_gain"] * 100

            # STCG scenario
            stcg_result = calculate_total_tax_stcg(income, total_gain, filing_status)
            stcg_total_tax[i, j] = stcg_result["total_tax"]
            stcg_federal_tax[i, j] = stcg_result["federal_stcg_tax"]
            stcg_ca_tax[i, j] = stcg_result["ca_tax"]
            stcg_niit[i, j] = stcg_result["niit"]
            stcg_effective_rate[i, j] = stcg_result["effective_rate_on_gain"] * 100

            # NIIT applicability
            magi = income + total_gain
            niit_applies[i, j] = 1 if magi > NIIT_THRESHOLD[filing_status] else 0

    # Tax savings from LTCG vs STCG
    tax_savings = stcg_total_tax - ltcg_total_tax

    return {
        "incomes": incomes,
        "prices": prices,
        "capital_gains": capital_gains,
        # LTCG components
        "ltcg_total_tax": ltcg_total_tax,
        "ltcg_federal_tax": ltcg_federal_tax,
        "ltcg_ca_tax": ltcg_ca_tax,
        "ltcg_niit": ltcg_niit,
        "ltcg_effective_rate": ltcg_effective_rate,
        # STCG components
        "stcg_total_tax": stcg_total_tax,
        "stcg_federal_tax": stcg_federal_tax,
        "stcg_ca_tax": stcg_ca_tax,
        "stcg_niit": stcg_niit,
        "stcg_effective_rate": stcg_effective_rate,
        # Derived
        "tax_savings": tax_savings,
        "niit_applies": niit_applies,
    }


def calculate_representative_scenarios(config: dict) -> pd.DataFrame:
    """Calculate detailed tax breakdown for specific scenarios defined in config."""
    scenarios = config.get("representative_scenarios", [
        {"income": 150_000, "price": 20},
        {"income": 250_000, "price": 30},
        {"income": 400_000, "price": 40},
    ])

    results = []
    filing_status = config["filing_status"]

    for scenario in scenarios:
        income = scenario["income"]
        price = scenario["price"]
        gain_per_share = price - config["cost_basis_per_share"]
        total_gain = gain_per_share * config["num_shares"]

        ltcg = calculate_total_tax_ltcg(income, total_gain, filing_status)
        stcg = calculate_total_tax_stcg(income, total_gain, filing_status)

        results.append({
            "Scenario": f"${income/1000:.0f}K income, ${price}/share",
            "Capital Gain": f"${total_gain:,.0f}",
            "LTCG Total Tax": f"${ltcg['total_tax']:,.0f}",
            "STCG Total Tax": f"${stcg['total_tax']:,.0f}",
            "Tax Savings (LTCG)": f"${stcg['total_tax'] - ltcg['total_tax']:,.0f}",
            "LTCG Fed Tax": f"${ltcg['federal_ltcg_tax']:,.0f}",
            "STCG Fed Tax": f"${stcg['federal_stcg_tax']:,.0f}",
            "NIIT": f"${ltcg['niit']:,.0f}",
            "CA Tax (LTCG)": f"${ltcg['ca_tax']:,.0f}",
            "Effective Rate (LTCG)": f"{ltcg['effective_rate_on_gain']*100:.1f}%",
            "Effective Rate (STCG)": f"{stcg['effective_rate_on_gain']*100:.1f}%",
        })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_dashboard(results: dict, config: dict) -> go.Figure:
    """Create interactive Plotly dashboard with multiple heatmaps and synchronized hover."""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "LTCG Total Tax Liability",
            "STCG Total Tax Liability",
            "Tax Savings (LTCG vs STCG)",
            "Effective Tax Rate on Capital Gain (LTCG)"
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    # Build customdata array with all tax components for unified hover display
    # Shape: (n_prices, n_incomes, 12) - storing all breakdown values
    n_prices = len(results["prices"])
    n_incomes = len(results["incomes"])

    # Calculate total sale proceeds for each price point
    # total_sale = price * num_shares (broadcast across income dimension)
    total_sale_proceeds = np.outer(results["prices"], np.ones(n_incomes)) * config["num_shares"]

    customdata = np.stack([
        results["capital_gains"] / 1000,        # 0: Capital gain ($K)
        results["ltcg_total_tax"] / 1000,       # 1: LTCG total ($K)
        results["ltcg_federal_tax"] / 1000,     # 2: LTCG federal ($K)
        results["ltcg_ca_tax"] / 1000,          # 3: LTCG CA ($K)
        results["ltcg_niit"] / 1000,            # 4: LTCG NIIT ($K)
        results["ltcg_effective_rate"],         # 5: LTCG eff rate (%)
        results["stcg_total_tax"] / 1000,       # 6: STCG total ($K)
        results["stcg_federal_tax"] / 1000,     # 7: STCG federal ($K)
        results["stcg_ca_tax"] / 1000,          # 8: STCG CA ($K)
        results["stcg_niit"] / 1000,            # 9: STCG NIIT ($K)
        results["stcg_effective_rate"],         # 10: STCG eff rate (%)
        results["tax_savings"] / 1000,          # 11: Savings ($K)
        total_sale_proceeds / 1000,             # 12: Total sale proceeds ($K)
    ], axis=-1)

    # Unified hover template showing all values
    unified_hover = (
        "<b>Income: $%{x:.0f}K | Price: $%{y:.0f}/share</b><br>"
        "<b>Total Sale: $%{customdata[12]:.1f}K</b> | <b>Capital Gain: $%{customdata[0]:.1f}K</b><br>"
        "<br>"
        "<b>LTCG Scenario:</b><br>"
        "  Federal: $%{customdata[2]:.1f}K<br>"
        "  CA State: $%{customdata[3]:.1f}K<br>"
        "  NIIT: $%{customdata[4]:.1f}K<br>"
        "  <b>Total: $%{customdata[1]:.1f}K</b> (%{customdata[5]:.1f}%)<br>"
        "<br>"
        "<b>STCG Scenario:</b><br>"
        "  Federal: $%{customdata[7]:.1f}K<br>"
        "  CA State: $%{customdata[8]:.1f}K<br>"
        "  NIIT: $%{customdata[9]:.1f}K<br>"
        "  <b>Total: $%{customdata[6]:.1f}K</b> (%{customdata[10]:.1f}%)<br>"
        "<br>"
        "<b>LTCG Savings: $%{customdata[11]:.1f}K</b>"
        "<extra></extra>"
    )

    # Heatmap 1: LTCG Total Tax
    fig.add_trace(
        go.Heatmap(
            z=results["ltcg_total_tax"] / 1000,
            x=results["incomes"] / 1000,
            y=results["prices"],
            customdata=customdata,
            colorscale="YlOrRd",
            colorbar=dict(title="Tax ($K)", x=0.45, len=0.4, y=0.8),
            hovertemplate=unified_hover,
            name="LTCG Tax",
        ),
        row=1, col=1
    )

    # Heatmap 2: STCG Total Tax
    fig.add_trace(
        go.Heatmap(
            z=results["stcg_total_tax"] / 1000,
            x=results["incomes"] / 1000,
            y=results["prices"],
            customdata=customdata,
            colorscale="YlOrRd",
            colorbar=dict(title="Tax ($K)", x=1.0, len=0.4, y=0.8),
            hovertemplate=unified_hover,
            name="STCG Tax",
        ),
        row=1, col=2
    )

    # Heatmap 3: Tax Savings
    fig.add_trace(
        go.Heatmap(
            z=results["tax_savings"] / 1000,
            x=results["incomes"] / 1000,
            y=results["prices"],
            customdata=customdata,
            colorscale="RdYlGn",
            colorbar=dict(title="Savings ($K)", x=0.45, len=0.4, y=0.2),
            hovertemplate=unified_hover,
            name="Tax Savings",
        ),
        row=2, col=1
    )

    # Heatmap 4: Effective Rate on Gain (LTCG)
    fig.add_trace(
        go.Heatmap(
            z=results["ltcg_effective_rate"],
            x=results["incomes"] / 1000,
            y=results["prices"],
            customdata=customdata,
            colorscale="Viridis",
            colorbar=dict(title="Rate (%)", x=1.0, len=0.4, y=0.2),
            hovertemplate=unified_hover,
            name="Eff. Rate",
        ),
        row=2, col=2
    )

    # Add NIIT threshold contour - clipped to valid positive income range
    niit_threshold = NIIT_THRESHOLD[config["filing_status"]]
    income_min = config["income_range"][0] / 1000
    income_max = config["income_range"][1] / 1000

    # Calculate contour where income + gain = NIIT threshold
    # Only include points where income is within the valid range
    niit_contour_prices = []
    niit_contour_incomes = []
    for price in results["prices"]:
        gain = (price - config["cost_basis_per_share"]) * config["num_shares"]
        income_at_threshold = (niit_threshold - gain) / 1000
        # Only include if income is within the plotted range
        if income_min <= income_at_threshold <= income_max:
            niit_contour_prices.append(price)
            niit_contour_incomes.append(income_at_threshold)

    # Add NIIT threshold line to each subplot (only if we have valid points)
    if niit_contour_incomes:
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_trace(
                go.Scatter(
                    x=niit_contour_incomes,
                    y=niit_contour_prices,
                    mode="lines",
                    line=dict(color="white", width=2, dash="dash"),
                    name="NIIT Threshold" if row == 1 and col == 1 else None,
                    showlegend=(row == 1 and col == 1),
                    hoverinfo="skip",
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Capital Gains Tax Sensitivity Analysis (2026)<br>" +
                 f"<sup>{config['num_shares']:,} shares | Cost basis: ${config['cost_basis_per_share']}/share | Filing: {config['filing_status']}</sup>",
            x=0.5,
            font=dict(size=18)
        ),
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="monospace",
        ),
    )

    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Ordinary Income ($K)", row=i, col=j)
            fig.update_yaxes(title_text="Sale Price ($/share)", row=i, col=j)

    return fig


def create_insights_table(results: dict, config: dict) -> str:
    """Generate key insights as HTML table."""

    filing_status = config["filing_status"]
    niit_threshold = NIIT_THRESHOLD[filing_status]

    # Find key breakpoints
    insights = []

    # NIIT breakpoint for various prices
    insights.append(("NIIT Threshold (MAGI)", f"${niit_threshold:,}"))

    for price in [10, 20, 30, 40, 50]:
        if price >= config["price_range"][0] and price <= config["price_range"][1]:
            gain = (price - config["cost_basis_per_share"]) * config["num_shares"]
            income_for_niit = niit_threshold - gain
            if income_for_niit > 0:
                insights.append(
                    (f"Income where NIIT kicks in @ ${price}/share",
                     f"${max(0, income_for_niit):,.0f}")
                )

    # Max tax savings point
    max_savings_idx = np.unravel_index(np.argmax(results["tax_savings"]),
                                        results["tax_savings"].shape)
    max_savings = results["tax_savings"][max_savings_idx]
    max_savings_price = results["prices"][max_savings_idx[0]]
    max_savings_income = results["incomes"][max_savings_idx[1]]

    insights.append(("Maximum LTCG Tax Savings", f"${max_savings:,.0f}"))
    insights.append(("  at Price/Income", f"${max_savings_price:.0f}/share, ${max_savings_income:,.0f} income"))

    # Federal LTCG bracket thresholds
    ltcg_brackets = (FEDERAL_LTCG_BRACKETS_SINGLE if filing_status == "single"
                     else FEDERAL_LTCG_BRACKETS_MARRIED)

    insights.append(("0% LTCG Threshold (taxable income)", f"${ltcg_brackets[0][0]:,}"))
    insights.append(("15% LTCG Threshold (taxable income)", f"${ltcg_brackets[1][0]:,}"))

    html = """
    <style>
        .insights-table {
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 20px 0;
        }
        .insights-table th, .insights-table td {
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: left;
        }
        .insights-table th {
            background-color: #4472C4;
            color: white;
        }
        .insights-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
    <table class="insights-table">
        <tr><th>Key Metric</th><th>Value</th></tr>
    """

    for metric, value in insights:
        html += f"<tr><td>{metric}</td><td>{value}</td></tr>"

    html += "</table>"

    return html


def create_scenarios_html(scenarios_df: pd.DataFrame) -> str:
    """Convert scenarios DataFrame to styled HTML."""
    html = """
    <style>
        .scenarios-table {
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 20px 0;
            font-size: 14px;
        }
        .scenarios-table th, .scenarios-table td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: right;
        }
        .scenarios-table th {
            background-color: #2E7D32;
            color: white;
            text-align: center;
        }
        .scenarios-table td:first-child {
            text-align: left;
            font-weight: bold;
        }
        .scenarios-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .scenarios-table tr:hover {
            background-color: #e8f5e9;
        }
    </style>
    """
    html += scenarios_df.to_html(classes="scenarios-table", index=False)
    return html


def generate_full_html_report(results: dict, config: dict,
                               scenarios_df: pd.DataFrame) -> str:
    """Generate complete HTML report with dashboard and tables."""

    fig = create_dashboard(results, config)
    dashboard_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    insights_html = create_insights_table(results, config)
    scenarios_html = create_scenarios_html(scenarios_df)

    # JavaScript for synchronized crosshair markers using SVG overlay (doesn't interfere with zoom)
    sync_hover_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            var graphDiv = document.querySelector('.plotly-graph-div');
            if (!graphDiv) return;

            // Create SVG overlay for crosshairs (doesn't interfere with Plotly interactions)
            var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1000;';
            svg.id = 'crosshair-overlay';
            graphDiv.style.position = 'relative';
            graphDiv.appendChild(svg);

            // Get subplot positions from Plotly layout
            function getSubplotBounds() {
                var layout = graphDiv._fullLayout;
                return [
                    { x: layout.xaxis, y: layout.yaxis },
                    { x: layout.xaxis2, y: layout.yaxis2 },
                    { x: layout.xaxis3, y: layout.yaxis3 },
                    { x: layout.xaxis4, y: layout.yaxis4 }
                ];
            }

            function drawCrosshairs(dataX, dataY) {
                svg.innerHTML = '';
                var bounds = getSubplotBounds();
                var plotWidth = graphDiv.clientWidth;
                var plotHeight = graphDiv.clientHeight;

                bounds.forEach(function(axes) {
                    if (!axes.x || !axes.y) return;

                    // Convert data coordinates to pixel coordinates for this subplot
                    var xDomain = axes.x.domain;
                    var yDomain = axes.y.domain;
                    var xRange = axes.x.range;
                    var yRange = axes.y.range;

                    // Subplot pixel bounds
                    var left = xDomain[0] * plotWidth;
                    var right = xDomain[1] * plotWidth;
                    var top = (1 - yDomain[1]) * plotHeight;
                    var bottom = (1 - yDomain[0]) * plotHeight;

                    // Data to pixel conversion
                    var xPx = left + (dataX - xRange[0]) / (xRange[1] - xRange[0]) * (right - left);
                    var yPx = bottom - (dataY - yRange[0]) / (yRange[1] - yRange[0]) * (bottom - top);

                    // Only draw if within subplot bounds
                    if (xPx >= left && xPx <= right && yPx >= top && yPx <= bottom) {
                        // Vertical line
                        var vLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        vLine.setAttribute('x1', xPx);
                        vLine.setAttribute('x2', xPx);
                        vLine.setAttribute('y1', top);
                        vLine.setAttribute('y2', bottom);
                        vLine.setAttribute('stroke', 'rgba(255,255,255,0.8)');
                        vLine.setAttribute('stroke-width', '1');
                        vLine.setAttribute('stroke-dasharray', '4,4');
                        svg.appendChild(vLine);

                        // Horizontal line
                        var hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        hLine.setAttribute('x1', left);
                        hLine.setAttribute('x2', right);
                        hLine.setAttribute('y1', yPx);
                        hLine.setAttribute('y2', yPx);
                        hLine.setAttribute('stroke', 'rgba(255,255,255,0.8)');
                        hLine.setAttribute('stroke-width', '1');
                        hLine.setAttribute('stroke-dasharray', '4,4');
                        svg.appendChild(hLine);
                    }
                });
            }

            graphDiv.on('plotly_hover', function(data) {
                var point = data.points[0];
                if (point.curveNumber > 3) return;
                drawCrosshairs(point.x, point.y);
            });

            graphDiv.on('plotly_unhover', function(data) {
                svg.innerHTML = '';
            });

            // Redraw on zoom/pan
            graphDiv.on('plotly_relayout', function() {
                svg.innerHTML = '';
            });

            console.log('SVG crosshair overlay initialized');
        }, 1000);
    });
    </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Capital Gains Tax Analysis - 2026</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            .disclaimer {{
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .assumptions {{
                background-color: #e3f2fd;
                border: 1px solid #2196f3;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .section {{
                margin: 30px 0;
            }}
            ul {{
                line-height: 1.8;
            }}
            /* Tooltip styling for better visibility */
            .hoverlayer .hovertext {{
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Capital Gains Tax Sensitivity Analysis</h1>
            <h3>US California Resident - Tax Year 2026 (Estimated Brackets)</h3>

            <div class="disclaimer">
                <strong>Disclaimer:</strong> This analysis is for educational and planning purposes only.
                Tax laws are complex and subject to change. Consult a qualified tax professional
                before making financial decisions.
            </div>

            <div class="assumptions">
                <strong>Analysis Parameters:</strong>
                <ul>
                    <li>Number of shares: {config['num_shares']:,}</li>
                    <li>Cost basis per share: ${config['cost_basis_per_share']}</li>
                    <li>Filing status: {config['filing_status'].title()}</li>
                    <li>Ordinary income range: ${config['income_range'][0]:,} - ${config['income_range'][1]:,}</li>
                    <li>Sale price range: ${config['price_range'][0]} - ${config['price_range'][1]} per share</li>
                </ul>
            </div>

            <div class="section">
                <h2>Interactive Dashboard</h2>
                <p>Hover over any heatmap to see a <strong>unified tooltip</strong> with full tax breakdown (Federal, CA State, NIIT) for both LTCG and STCG scenarios.
                   Crosshairs synchronize across all four plots. The dashed white line shows where NIIT (3.8%) begins to apply.</p>
                {dashboard_html}
            </div>

            <div class="section">
                <h2>Key Insights & Breakpoints</h2>
                {insights_html}
            </div>

            <div class="section">
                <h2>Representative Scenarios</h2>
                {scenarios_html}
            </div>

            <div class="section">
                <h2>Tax Calculation Notes</h2>
                <ul>
                    <li><strong>LTCG (Long-Term Capital Gains):</strong> Applies when shares are held >1 year.
                        Federal rates: 0%/15%/20% based on taxable income brackets.</li>
                    <li><strong>STCG (Short-Term Capital Gains):</strong> Applies when shares are held ≤1 year.
                        Taxed as ordinary income at federal level (10-37%).</li>
                    <li><strong>California:</strong> Taxes ALL capital gains as ordinary income (1-13.3%).
                        No preferential rate for LTCG at state level.</li>
                    <li><strong>NIIT:</strong> 3.8% Net Investment Income Tax applies when MAGI exceeds
                        ${NIIT_THRESHOLD['single']:,} (single) or ${NIIT_THRESHOLD['married']:,} (married).</li>
                    <li><strong>AMT:</strong> Not modeled. May apply in some scenarios, especially with ISO exercises.</li>
                </ul>
            </div>
        </div>
        {sync_hover_js}
    </body>
    </html>
    """

    return html


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(config_path: str = None):
    """
    Run the full analysis and generate outputs.

    Args:
        config_path: Optional path to config JSON file. If not provided,
                     will look for config.json then config.example.json.
    """
    # Load configuration
    config = load_config(config_path)

    print("=" * 60)
    print("Capital Gains Tax Sensitivity Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Shares: {config['num_shares']:,}")
    print(f"  Cost basis: ${config['cost_basis_per_share']}/share")
    print(f"  Filing status: {config['filing_status']}")
    print(f"  Income range: ${config['income_range'][0]:,} - ${config['income_range'][1]:,}")
    print(f"  Price range: ${config['price_range'][0]} - ${config['price_range'][1]}/share")

    # Run analysis
    print("\nRunning sensitivity analysis...")
    results = run_sensitivity_analysis(config)

    # Generate representative scenarios
    print("Calculating representative scenarios...")
    scenarios_df = calculate_representative_scenarios(config)
    print("\nRepresentative Scenarios:")
    print(scenarios_df.to_string(index=False))

    # Generate HTML report
    print("\nGenerating HTML report...")
    html_report = generate_full_html_report(results, config, scenarios_df)

    # Save outputs
    output_file = "tax_analysis_dashboard.html"
    with open(output_file, "w") as f:
        f.write(html_report)
    print(f"\nSaved interactive dashboard: {output_file}")

    # Save individual figure as HTML
    fig = create_dashboard(results, config)
    fig.write_html("tax_heatmaps.html")
    print("Saved heatmaps: tax_heatmaps.html")

    # Save static images
    try:
        fig.write_image("tax_heatmaps.png", scale=2)
        print("Saved static image: tax_heatmaps.png")
    except Exception as e:
        print(f"Note: Could not save PNG (install kaleido: pip install kaleido). Error: {e}")

    # Save scenarios to CSV
    scenarios_df.to_csv("representative_scenarios.csv", index=False)
    print("Saved scenarios: representative_scenarios.csv")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return results, scenarios_df


if __name__ == "__main__":
    # Support optional command-line argument for config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    results, scenarios = main(config_path)
