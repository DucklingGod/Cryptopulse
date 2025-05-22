"""
Fetch Bitcoin social topic summary from LunarCrush and print insights.
"""
import requests

API_URL = "https://lunarcrush.com/api4/public/topic/bitcoin/v1"
HEADERS = {"Authorization": "Bearer undefined"}

response = requests.get(API_URL, headers=HEADERS)
if response.status_code != 200:
    print(f"[ERROR] Failed to fetch data: {response.status_code} {response.text}")
    exit(1)

data = response.json().get("data", {})
if not data:
    print("No summary data found.")
    exit(0)

print(f"Bitcoin Social Topic Summary (LunarCrush):\n")
print(f"Title: {data.get('title')}")
print(f"Topic Rank: {data.get('topic_rank')}")
print(f"Categories: {', '.join(data.get('categories', []))}")
print(f"Trend: {data.get('trend').capitalize() if data.get('trend') else 'N/A'}")
print(f"Interactions (24h): {data.get('interactions_24h'):,}")
print(f"Contributors (24h): {data.get('num_contributors'):,}")
print(f"Posts (24h): {data.get('num_posts'):,}\n")

# 1. Most active social type by post count
most_active_type = max(data['types_count'], key=lambda k: data['types_count'][k])
print(f"Most Active Social Type: {most_active_type} ({data['types_count'][most_active_type]:,} posts)")

# 2. Social type with most interactions
most_interactive_type = max(data['types_interactions'], key=lambda k: data['types_interactions'][k])
print(f"Most Interactive Social Type: {most_interactive_type} ({data['types_interactions'][most_interactive_type]:,} interactions)")

# 3. Social type with highest sentiment
most_positive_type = max(data['types_sentiment'], key=lambda k: data['types_sentiment'][k])
print(f"Most Positive Social Type: {most_positive_type} ({data['types_sentiment'][most_positive_type]}% positive)\n")

# 4. Sentiment breakdown for each type
print("Sentiment Breakdown by Social Type:")
for stype, detail in data['types_sentiment_detail'].items():
    total = sum(detail.values())
    pos = detail.get('positive', 0)
    neu = detail.get('neutral', 0)
    neg = detail.get('negative', 0)
    print(f"  {stype}: {pos:,} positive, {neu:,} neutral, {neg:,} negative (Total: {total:,})")

# 5. Related topics
print(f"\nRelated Topics: {', '.join(data.get('related_topics', []))}")
