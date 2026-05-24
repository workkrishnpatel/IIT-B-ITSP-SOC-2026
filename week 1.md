
# Advisory Memo: Downside Portfolio Hedging Strategy

**Prepared For:** Wealth Management Client  
**Prepared By:** Krishna Patel  
**Date:** May 24, 2026  
**Subject:** Downside Hedging of Equity Portfolio Using Protective Put Options  



## Executive Summary

The purpose of this memo is to evaluate, structure, and recommend a derivative overlay strategy to insulate an existing large-cap equity position from a projected short-term market contraction. 

The client holds **500 shares** of a high-conviction large-cap stock currently trading at **₹2,450**. Anticipating a **10% to 15% correction** over the next 3 months, the client requires a risk-mitigation framework that preserves long-term equity ownership, avoids triggering immediate capital gains tax liabilities, and caps structural downside losses.



## Client Risk Profile & Scenario Analysis

### Current Position Metrics
* **Asset Class:** Large-Cap Equity
* **Position Size ($N$):** 500 shares
* **Current Spot Price ($S_0$):** ₹2,450 per share
* **Total Portfolio Market Value ($V_0$):** $$V_0 = 500 \times 2,450 = ₹1,225,000$$

### Unhedged Worst-Case Scenario (15% Market Correction)
If the market undergoes a maximum expected correction of 15% over the next 90 days, the terminal spot price ($S_T$) will decline to **₹2,082.50**.

* **Terminal Stock Price ($S_T$):** $$S_T = 2,450 \times (1 - 0.15) = ₹2,082.50$$
* **Absolute Portfolio Loss (Unhedged):** $$\text{Loss} = N \times (S_0 - S_T) = 500 \times (2,450 - 2,082.50) = ₹183,750$$

Without a structural derivative overlay, the portfolio value will contract directly from ₹1,225,000 to ₹1,041,250.



## Derivative Instrument Selection: Call vs. Put Options

To offset the calculated ₹183,750 exposure, we analyze the performance profiles of standard European options under a downward price trajectory.

### 1. Long Call Option Profile
The mathematical payoff for a long call option at expiration is defined as:
$$\text{Payoff}_{\text{Call}} = \max(S_T - K, 0)$$

Where $K$ represents the chosen contract strike price. Under the expected downside scenario where $S_T = ₹2,082.50$, selecting a strike price equal to or greater than the current spot ($K \ge ₹2,450$) results in:
$$\text{Payoff}_{\text{Call}} = \max(2,082.50 - 2,450, 0) = ₹0$$

* **Strategic Evaluation:** Long call options provide zero capital compensation during a falling market, making them completely ineffective for downside insulation.

### 2. Long Put Option Profile (Protective Put)
The mathematical payoff for a long put option at expiration is defined as:
$$\text{Payoff}_{\text{Put}} = \max(K - S_T, 0)$$

As the terminal stock price ($S_T$) drops below the strike price ($K$), the value of the put option contract grows at a 1:1 linear rate relative to the underlying asset's decline.

* **Strategic Evaluation:** The intrinsic value accumulation directly counters the loss on the underlying spot position. This makes the **Protective Put Strategy** the ideal mechanic for structural portfolio preservation.



## Hedging Mechanics & Payoff Modeling

To balance protection with the cost of hedging (premium), a put option contract with a **strike price ($K$) of ₹2,300** and a **3-month expiration** is selected.

### Performance Modeling ($S_T = ₹2,082.50$)

#### 1. Underlying Equity Position Value
$$V_{\text{Equity}} = 500 \times 2,082.50 = ₹1,041,250$$
$$\Delta V_{\text{Equity}} = -₹183,750$$

#### 2. Long Put Option Derivative Position
$$\text{Per-Share Intrinsic Payoff} = \max(2,300 - 2,082.50, 0) = ₹217.50$$
$$\text{Total Gross Derivative Gain} = 500 \times 217.50 = +₹108,750$$

#### 3. Combined Hedged Portfolio Value
$$\text{Net Position Payoff} = \Delta V_{\text{Equity}} + \text{Total Gross Derivative Gain}$$
$$\text{Net Position Payoff} = -183,750 + 108,750 = -₹75,000$$



## Financial Visualizations

The target behavior of the individual components and the combined hedged portfolio are broken down across three main mathematical zones:

| Underlying Price Zone | Long Equity Vector | Long Put Option ($K=2300$) | Net Hedged Portfolio Profile |
| :--- | :--- | :--- | :--- |
| **Bullish ($S_T > ₹2,300$)** | Captures full linear upside and downside exposure. | Contract expires worthless; value resets flat to ₹0. | Tracks equity performance minus the option premium paid. |
| **At-The-Money ($S_T = ₹2,300$)** | Value sits down exactly at ₹2,300 per share. | Contract is at the money; value sits at ₹0. | Marks the exact point where the structural loss floor is triggered. |
| **Bearish ($S_T < ₹2,300$)** | Experiences continuous linear capital erosion. | Gains intrinsic value at a 1:1 rate below ₹2,300. | Portfolio value flattens completely; total losses are capped at ₹75,000. |



## Final Recommendation & Conclusion

A raw 15% market contraction would expose the client to a direct capital reduction of **₹183,750**. Liquidating the underlying shares to prevent this loss is counterproductive due to long-term investment conviction and the immediate trigger of capital gains tax.

Implementing the **Protective Put Hedging Strategy** with a strike price of ₹2,300 solves these conflicting constraints:

* **Downside Insulation:** Caps maximum risk at **₹75,000**, saving **₹108,750** in capital value that would otherwise be lost during the correction.
* **Tax and Ownership Preservation:** Avoids selling the underlying equity, preserving long-term voting rights and compounding advantages while deferring tax liabilities.
* **Upside Flexibility:** Retains the ability to profit from future upward movements once the temporary 3-month market correction passes.

**Action Item:** Open an options overlay window to acquire 3-month European put option contracts covering 500 underlying units at the ₹2,300 strike boundary.
