## Practice: Quantitative Trading Strategy

Here are mainly four sections in this particular project, which is a just simple practice.

#### Composition

I strongly suggest you to read this **<u>first</u>**, where I have introduced the whole outline of the project process.

#### Random Forest

It gives an probable strategy to select stocks which has **higher alpha($\alpha$)**. Stocks with higher alpha often perform better than others. We use a simple random forest model to predict higher alpha stocks.

#### XGBoost

Beyond **select stocks**, **market timing** is also the target of a multitude of arrows. I have construct a simple **XGBoost model** to predict stocks price using lagged, moving average, and market prices. Instead of sectional regression in *<u>Gu et al. (2020)</u> RFS paper*, I set models for each selected stock solely. The predict price and real price's contrast has been illustrated thoroughly in the *COMPOSITION* file. 

#### Inernational

In this section, I have used linear programming methods to construct investment portfolio. Accounting to Markowitz Mean-Variance Portfolio Model, it is valid to diversify your portfolio into different assets or different companies. In this part, I have construct an international investment strategy. Different from the previous methods, this part there is no difficult predict model. I have simply assume that $\hat r_{i,t+1} = E(r_{i,t}) = \overline r_{i,t}$, and respectively $\hat \sigma_{i,t+1} = Var(r_{i,t})$.


