"""
Fetch Bitcoin asset metrics from LunarCrush /v2 endpoint and print insights.
"""
import requests

API_KEY = "juitj7xt2wmoxfi8m01jpwich3maxjn93x214toim"
API_URL = f"https://api.lunarcrush.com/v2?data=assets&key={API_KEY}&symbol=BTC"

response = requests.get(API_URL)
if response.status_code != 200:
    print(f"[ERROR] Failed to fetch data: {response.status_code} {response.text}")
    exit(1)

data = response.json().get("data", [])
if not data:
    print("No asset data found.")
    exit(0)

btc = data[0]
print("LunarCrush BTC Asset Metrics (24h):\n")
print(f"Symbol: {btc.get('symbol')}")
print(f"Name: {btc.get('name')}")
print(f"Price: ${btc.get('price', 0):,.2f}")
print(f"Market Cap: ${btc.get('market_cap', 0):,.0f}")
print(f"Volume (24h): ${btc.get('volume', 0):,.0f}")
print(f"Social Volume: {btc.get('social_volume', 0):,}")
print(f"Social Score: {btc.get('social_score', 0):,}")
print(f"Average Sentiment: {btc.get('average_sentiment', 0):.2f}")
print(f"Galaxy Score: {btc.get('galaxy_score', 0):.2f}")
print(f"AltRank: {btc.get('alt_rank', 0)}")
print(f"Social Dominance: {btc.get('social_dominance', 0):.2f}%")
print(f"Social Contributors: {btc.get('social_contributors', 0):,}")
print(f"Social Engagement: {btc.get('social_engagement', 0):,}")
print(f"Social Mentions: {btc.get('social_mentions', 0):,}")
print(f"Social Posts: {btc.get('social_posts', 0):,}")
print(f"News: {btc.get('news', 0):,}")
print(f"URL: {btc.get('url', 'N/A')}")
