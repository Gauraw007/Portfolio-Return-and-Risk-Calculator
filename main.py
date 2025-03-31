# Portfolio Return and Risk Calculator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns

# Step 1: Define stocks and download historical price data
def download_stock_data(tickers, start_date, end_date):
    """Download historical stock data for the given tickers."""
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    # If only one ticker is downloaded, convert the Series to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data

# Step 2: Calculate daily returns
def calculate_returns(prices):
    """Calculate daily returns from price data."""
    returns = prices.pct_change().dropna()
    return returns

# Step 3: Calculate portfolio statistics with equal weights
def calculate_portfolio_stats(returns, weights=None):
    """
    Calculate portfolio return and risk statistics.
    
    Parameters:
    returns (DataFrame): Daily returns for each stock
    weights (array-like, optional): Portfolio weights. If None, equal weighting is used.
    
    Returns:
    dict: Dictionary containing portfolio statistics
    """
    if weights is None:
        weights = np.ones(returns.shape[1]) / returns.shape[1]  # Equal weights
    
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Calculate expected returns (annualized)
    expected_returns = np.sum(returns.mean() * weights) * 252
    
    # Calculate portfolio variance using the covariance matrix
    cov_matrix = returns.cov() * 252  # Annualized covariance
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = expected_returns / portfolio_volatility
    
    # Calculate individual stock statistics (annualized)
    individual_expected_returns = returns.mean() * 252
    individual_volatility = returns.std() * np.sqrt(252)
    
    return {
        'portfolio_return': expected_returns,
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'weights': weights,
        'cov_matrix': cov_matrix,
        'individual_returns': individual_expected_returns,
        'individual_volatility': individual_volatility
    }

# Step 4: Plot the results
def plot_portfolio_results(stats, tickers):
    """Create visualizations for portfolio analysis."""
    # Set plotting style
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 10))
    
    # 1. Risk-Return Scatter Plot for individual stocks and portfolio
    plt.subplot(2, 2, 1)
    # Plot individual stocks
    for i, ticker in enumerate(tickers):
        plt.scatter(
            stats['individual_volatility'][i], 
            stats['individual_returns'][i], 
            s=100, 
            label=ticker
        )
    
    # Plot portfolio
    plt.scatter(
        stats['portfolio_volatility'], 
        stats['portfolio_return'], 
        s=200, 
        color='red', 
        marker='*', 
        label='Portfolio'
    )
    
    plt.title('Risk-Return Profile')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Expected Return')
    plt.legend()
    plt.grid(True)
    
    # 2. Portfolio Weights Pie Chart
    plt.subplot(2, 2, 2)
    plt.pie(stats['weights'], labels=tickers, autopct='%1.1f%%')
    plt.title('Portfolio Allocation')
    
    # 3. Correlation Heatmap
    plt.subplot(2, 2, 3)
    correlation_matrix = stats['cov_matrix'] / (np.outer(np.sqrt(np.diag(stats['cov_matrix'])), np.sqrt(np.diag(stats['cov_matrix']))))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=tickers, yticklabels=tickers)
    plt.title('Stock Correlation Matrix')
    
    # 4. Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = (
        f"Portfolio Summary:\n\n"
        f"Expected Annual Return: {stats['portfolio_return']*100:.2f}%\n"
        f"Annual Volatility: {stats['portfolio_volatility']*100:.2f}%\n"
        f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n\n"
        f"Individual Stock Statistics:\n"
    )
    
    for i, ticker in enumerate(tickers):
        summary_text += (
            f"{ticker}:\n"
            f"  - Expected Return: {stats['individual_returns'][i]*100:.2f}%\n"
            f"  - Volatility: {stats['individual_volatility'][i]*100:.2f}%\n"
        )
    
    plt.text(0, 1, summary_text, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

# Step 5: Monte Carlo Simulation for portfolio optimization (extension)
def monte_carlo_simulation(returns, num_portfolios=10000):
    """
    Perform Monte Carlo simulation to find efficient frontier.
    
    Parameters:
    returns (DataFrame): Daily returns for each stock
    num_portfolios (int): Number of random portfolios to generate
    
    Returns:
    DataFrame: Results of simulation with returns, volatility, and weights
    """
    print("Running Monte Carlo simulation...")
    num_assets = returns.shape[1]
    results = np.zeros((num_portfolios, num_assets + 3))  # +3 for return, volatility, and Sharpe ratio
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(
            weights.T @ (returns.cov() * 252) @ weights
        )
        
        # Calculate Sharpe Ratio
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        # Store results
        results[i, 0] = portfolio_return
        results[i, 1] = portfolio_volatility
        results[i, 2] = sharpe_ratio
        results[i, 3:] = weights
    
    # Convert results to DataFrame
    columns = ['return', 'volatility', 'sharpe_ratio'] + list(returns.columns)
    results_df = pd.DataFrame(results, columns=columns)
    
    return results_df

def plot_efficient_frontier(results_df, stats, tickers):
    """Plot the efficient frontier from Monte Carlo simulation results."""
    plt.figure(figsize=(12, 8))
    
    # Plot all random portfolios
    plt.scatter(
        results_df['volatility'], 
        results_df['return'],
        c=results_df['sharpe_ratio'],
        cmap='viridis',
        alpha=0.3,
        s=10
    )
    
    # Mark current portfolio
    plt.scatter(
        stats['portfolio_volatility'],
        stats['portfolio_return'],
        c='red',
        marker='*',
        s=200,
        label='Current Portfolio'
    )
    
    # Find and mark optimal portfolio (highest Sharpe ratio)
    max_sharpe_idx = results_df['sharpe_ratio'].idxmax()
    plt.scatter(
        results_df.loc[max_sharpe_idx, 'volatility'],
        results_df.loc[max_sharpe_idx, 'return'],
        c='green',
        marker='X',
        s=200,
        label='Max Sharpe Ratio Portfolio'
    )
    
    # Find and mark minimum volatility portfolio
    min_vol_idx = results_df['volatility'].idxmin()
    plt.scatter(
        results_df.loc[min_vol_idx, 'volatility'],
        results_df.loc[min_vol_idx, 'return'],
        c='orange',
        marker='P',
        s=200,
        label='Min Volatility Portfolio'
    )
    
    # Plot individual stocks
    for i, ticker in enumerate(tickers):
        plt.scatter(
            stats['individual_volatility'][i],
            stats['individual_returns'][i],
            s=100,
            label=ticker
        )
    
    # Add colorbar for Sharpe ratio
    cbar = plt.colorbar()
    cbar.set_label('Sharpe Ratio')
    
    plt.title('Efficient Frontier')
    plt.xlabel('Annualized Volatility (Risk)')
    plt.ylabel('Annualized Expected Return')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print optimal portfolio weights
    print("\nOptimal Portfolio Weights (Maximum Sharpe Ratio):")
    optimal_weights = results_df.iloc[max_sharpe_idx, 3:]
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight*100:.2f}%")
    
    print(f"\nOptimal Portfolio Expected Return: {results_df.loc[max_sharpe_idx, 'return']*100:.2f}%")
    print(f"Optimal Portfolio Volatility: {results_df.loc[max_sharpe_idx, 'volatility']*100:.2f}%")
    print(f"Optimal Portfolio Sharpe Ratio: {results_df.loc[max_sharpe_idx, 'sharpe_ratio']:.2f}")

# Main function to run the analysis
def main():
    # Define portfolio parameters
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Example tickers
    
    # You can customize these parameters
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    weights = None  # Equal weighting by default, change if needed
    
    # Step 1: Download data
    prices = download_stock_data(tickers, start_date, end_date)
    print(f"Data downloaded successfully. Shape: {prices.shape}")
    
    # Display first few rows of price data
    print("\nPrice Data (first 5 rows):")
    print(prices.head())
    
    # Step 2: Calculate returns
    returns = calculate_returns(prices)
    print("\nReturns Data (first 5 rows):")
    print(returns.head())
    
    # Step 3: Calculate portfolio statistics
    stats = calculate_portfolio_stats(returns, weights)
    print("\nPortfolio Statistics:")
    print(f"Expected Annual Return: {stats['portfolio_return']*100:.2f}%")
    print(f"Annual Volatility: {stats['portfolio_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    # Step 4: Plot the results
    plot_portfolio_results(stats, tickers)
    
    # Step 5: Monte Carlo Simulation (extension)
    mc_results = monte_carlo_simulation(returns, num_portfolios=5000)
    plot_efficient_frontier(mc_results, stats, tickers)

if __name__ == "__main__":
    main()
