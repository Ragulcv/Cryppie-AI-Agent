import os
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CryptoAdvisor:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.market_url = f"{self.base_url}/coins/markets"
        try:
            self.analyzer = pipeline(
                'text-generation',
                model='EleutherAI/gpt-neo-125M',
                framework="pt"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.analyzer = None

    def get_numeric_input(self, prompt, allow_percentage=False):
        """Helper function to get numeric input with optional percentage handling"""
        while True:
            try:
                value = input(prompt).strip().lower()
                if allow_percentage and value.endswith('%'):
                    return float(value.rstrip('%'))
                return float(value)
            except ValueError:
                print("Please enter a valid number.")

    def get_basic_preferences(self):
        """Get basic user preferences for simple mode"""
        preferences = {}
        print("\n=== Basic Investment Parameters ===")
        preferences['investment_size'] = self.get_numeric_input("Enter investment size in USD: ")
        
        while True:
            risk = input("Risk profile (low/medium/high): ").lower()
            if risk in ['low', 'medium', 'high']:
                preferences['risk_profile'] = risk
                break
            print("Please enter 'low', 'medium', or 'high'")

        preferences['investment_horizon'] = int(self.get_numeric_input("Investment horizon (in months): "))
        preferences['expected_returns'] = self.get_numeric_input("Expected returns (%): ", allow_percentage=True)

        # Set default values for basic mode
        preferences['market_cap_preference'] = 'all'
        preferences['max_per_token'] = 30.0
        preferences['min_volume'] = 1000000.0
        preferences['stop_loss'] = 15.0
        preferences['take_profit'] = 30.0
        preferences['consider_volatility'] = True
        preferences['momentum_importance'] = 7
        preferences['volume_importance'] = 6

        return preferences

    def get_pro_preferences(self):
        """Get detailed user preferences for pro mode"""
        preferences = self.get_basic_preferences()
        
        print("\n=== Advanced Parameters (Pro Mode) ===")
        print("\n=== Market Preferences ===")
        while True:
            cap = input("Preferred market cap focus (large/mid/small/all): ").lower()
            if cap in ['large', 'mid', 'small', 'all']:
                preferences['market_cap_preference'] = cap
                break
            print("Please enter 'large', 'mid', 'small', or 'all'")

        preferences['max_per_token'] = self.get_numeric_input("Maximum allocation per token (%): ", allow_percentage=True)
        preferences['min_volume'] = self.get_numeric_input("Minimum 24h trading volume in USD: ")
        
        print("\n=== Risk Management ===")
        preferences['stop_loss'] = self.get_numeric_input("Default stop-loss percentage: ", allow_percentage=True)
        preferences['take_profit'] = self.get_numeric_input("Default take-profit percentage: ", allow_percentage=True)
        
        print("\n=== Technical Factors ===")
        preferences['consider_volatility'] = input("Consider volatility in selection? (yes/no): ").lower() == 'yes'
        preferences['momentum_importance'] = int(self.get_numeric_input("Importance of price momentum (1-10): "))
        preferences['volume_importance'] = int(self.get_numeric_input("Importance of trading volume (1-10): "))
        
        return preferences

    def fetch_market_data(self, preferences):
        """Fetch comprehensive market data based on user preferences"""
        try:
            # Fetch current market data
            current_params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "1h,24h,7d,30d"
            }
            
            response = requests.get(self.market_url, params=current_params)
            response.raise_for_status()
            market_data = response.json()
            
            # Filter based on user preferences
            filtered_data = []
            for coin in market_data:
                if (coin.get('total_volume', 0) >= preferences['min_volume'] and
                    self.matches_market_cap_preference(coin, preferences['market_cap_preference'])):
                    # Calculate additional metrics
                    coin['volatility_score'] = self.calculate_volatility_score(coin)
                    coin['momentum_score'] = self.calculate_momentum_score(coin)
                    coin['volume_score'] = self.calculate_volume_score(coin)
                    coin['overall_score'] = self.calculate_overall_score(coin, preferences)
                    filtered_data.append(coin)
            
            return filtered_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return []

    def matches_market_cap_preference(self, coin, preference):
        """Check if coin matches market cap preference"""
        market_cap = coin.get('market_cap', 0)
        if preference == 'all':
            return True
        elif preference == 'large' and market_cap >= 1e9:
            return True
        elif preference == 'mid' and 2e8 <= market_cap < 1e9:
            return True
        elif preference == 'small' and market_cap < 2e8:
            return True
        return False

    def calculate_volatility_score(self, coin):
        """Calculate volatility score based on price changes"""
        changes = [
            abs(float(coin.get('price_change_percentage_1h_in_currency', 0) or 0)),
            abs(float(coin.get('price_change_percentage_24h_in_currency', 0) or 0)),
            abs(float(coin.get('price_change_percentage_7d_in_currency', 0) or 0))
        ]
        return np.mean(changes) if changes else 0

    def calculate_momentum_score(self, coin):
        """Calculate momentum score based on price trends"""
        changes = [
            float(coin.get('price_change_percentage_24h_in_currency', 0) or 0),
            float(coin.get('price_change_percentage_7d_in_currency', 0) or 0),
            float(coin.get('price_change_percentage_30d_in_currency', 0) or 0)
        ]
        return np.mean(changes) if changes else 0

    def calculate_volume_score(self, coin):
        """Calculate volume score based on trading activity"""
        market_cap = float(coin.get('market_cap', 0))
        volume = float(coin.get('total_volume', 0))
        return (volume / market_cap) if market_cap > 0 else 0

    def calculate_overall_score(self, coin, preferences):
        """Calculate overall score based on user preferences"""
        scores = {
            'momentum': coin['momentum_score'] * preferences['momentum_importance'] / 10,
            'volume': coin['volume_score'] * preferences['volume_importance'] / 10
        }
        
        if preferences['consider_volatility']:
            scores['volatility'] = coin['volatility_score']
        
        return np.mean(list(scores.values()))

    def visualize_portfolio(self, portfolio):
        """Create and display portfolio visualization"""
        if not portfolio['allocations']:
            return

        # Prepare data for pie chart
        labels = [f"{alloc['symbol'].upper()}\n{alloc['percentage']:.1f}%" 
                 for alloc in portfolio['allocations']]
        sizes = [alloc['percentage'] for alloc in portfolio['allocations']]
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Recommended Portfolio Allocation')
        
        # Add legend
        plt.legend(labels, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        try:
            plt.savefig('portfolio_allocation.png', bbox_inches='tight')
            print("\nPortfolio visualization saved as 'portfolio_allocation.png'")
        except Exception as e:
            print(f"\nError saving visualization: {e}")
        
        plt.close()

    def generate_portfolio_recommendation(self, market_data, preferences):
        """Generate portfolio recommendation using market data and preferences"""
        try:
            # Sort coins by overall score
            sorted_coins = sorted(market_data, key=lambda x: x['overall_score'], reverse=True)
            
            # Select top coins based on risk profile
            num_coins = {
                'low': 3,
                'medium': 5,
                'high': 7
            }.get(preferences['risk_profile'], 5)
            
            selected_coins = sorted_coins[:num_coins]
            
            # Calculate allocations
            total_score = sum(coin['overall_score'] for coin in selected_coins)
            allocations = []
            
            for coin in selected_coins:
                percentage = (coin['overall_score'] / total_score) * 100
                if percentage > preferences['max_per_token']:
                    percentage = preferences['max_per_token']
                
                amount = (percentage / 100) * preferences['investment_size']
                
                allocations.append({
                    'name': coin['name'],
                    'symbol': coin['symbol'],
                    'percentage': percentage,
                    'amount': amount
                })
            
            # Normalize percentages if needed
            total_percentage = sum(a['percentage'] for a in allocations)
            if total_percentage != 100:
                for allocation in allocations:
                    allocation['percentage'] = (allocation['percentage'] / total_percentage) * 100
                    allocation['amount'] = (allocation['percentage'] / 100) * preferences['investment_size']
            
            # Generate portfolio strategy
            strategy = {
                'allocations': allocations,
                'rationale': self.generate_rationale(allocations, preferences),
                'risk_management': self.generate_risk_management(preferences),
                'rebalancing': self.generate_rebalancing_strategy(preferences)
            }
            
            return strategy
            
        except Exception as e:
            print(f"Error generating portfolio recommendation: {e}")
            return None

    def generate_rationale(self, allocations, preferences):
        """Generate investment rationale based on allocations and preferences"""
        risk_levels = {
            'low': 'conservative',
            'medium': 'balanced',
            'high': 'aggressive'
        }
        
        rationale = (
            f"This {risk_levels[preferences['risk_profile']]} portfolio is designed for a "
            f"{preferences['investment_horizon']}-month investment horizon with an expected "
            f"return of {preferences['expected_returns']}%.\n\n"
        )
        
        for allocation in allocations:
            rationale += (
                f"{allocation['name']} ({allocation['symbol'].upper()}): "
                f"Allocated {allocation['percentage']:.1f}% due to strong "
                f"performance metrics and market position.\n"
            )
            
        return rationale

    def generate_risk_management(self, preferences):
        """Generate risk management strategy"""
        return (
            f"Stop-loss set at {preferences['stop_loss']}% to limit potential losses.\n"
            f"Take-profit targets set at {preferences['take_profit']}% to secure gains.\n"
            "Regular monitoring and rebalancing recommended to maintain target allocations."
        )

    def generate_rebalancing_strategy(self, preferences):
        """Generate rebalancing strategy based on preferences"""
        if preferences['investment_horizon'] <= 3:
            interval = "monthly"
        elif preferences['investment_horizon'] <= 6:
            interval = "bi-monthly"
        else:
            interval = "quarterly"
            
        return (
            f"Recommended rebalancing interval: {interval}\n"
            "Review and adjust allocations based on:\n"
            "- Market conditions and trend changes\n"
            "- Individual asset performance\n"
            "- Risk tolerance changes"
        )

def main():
    advisor = CryptoAdvisor()
    
    print("=== Cryptocurrency Investment Advisor ===")
    
    # Ask user for mode preference
    while True:
        mode = input("\nChoose mode (basic/pro): ").lower()
        if mode in ['basic', 'pro']:
            break
        print("Please enter 'basic' or 'pro'")
    
    # Get preferences based on mode
    preferences = advisor.get_pro_preferences() if mode == 'pro' else advisor.get_basic_preferences()
    
    # Fetch market data
    print("\nAnalyzing market data...")
    market_data = advisor.fetch_market_data(preferences)
    
    if not market_data:
        print("Error: Unable to fetch market data. Please try again later.")
        return
    
    # Generate portfolio recommendation
    print("Generating portfolio recommendation...")
    portfolio = advisor.generate_portfolio_recommendation(market_data, preferences)
    
    if portfolio:
        print("\n=== Investment Portfolio ===")
        print("\nAllocations:")
        for allocation in portfolio['allocations']:
            print(f"- {allocation['name']} ({allocation['symbol'].upper()}): "
                  f"{allocation['percentage']:.1f}% (${allocation['amount']:,.2f})")
        
        # Create visualization
        advisor.visualize_portfolio(portfolio)
        
        print("\nRationale:")
        print(portfolio['rationale'])
        
        print("\nRisk Management Strategy:")
        print(portfolio['risk_management'])
        
        if mode == 'pro':
            print("\nRebalancing Strategy:")
            print(portfolio['rebalancing'])
    else:
        print("Error generating portfolio recommendation.")

if __name__ == "__main__":
    main()