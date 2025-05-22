"""
Daily pipeline entry point for Google Cloud Run Job.
This script runs all steps needed to fetch, process, and merge social sentiment and price data for BTC.
"""

def main():
    # Import and run each pipeline step
    from exorde_sentiment_insights import main as exorde_main
    from merge_sentiment_with_price import main as merge_main
    from append_realtime_exorde_to_merged import main as append_exorde_main
    # Add more steps as needed (e.g., lunarcrush, ML retrain, etc.)

    print("[PIPELINE] Running Exorde sentiment fetch...")
    exorde_main()
    print("[PIPELINE] Merging sentiment with price data...")
    merge_main()
    print("[PIPELINE] Appending real-time Exorde to merged data...")
    append_exorde_main()
    print("[PIPELINE] Daily pipeline completed.")

if __name__ == "__main__":
    main()
