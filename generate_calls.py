"""Generate fake Aircall call data for demonstration."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Call topics and templates
TOPICS = {
    "billing": [
        "Customer called about incorrect billing charges on their account.",
        "Billing inquiry regarding invoice discrepancies. Resolved by adjusting charges.",
        "Customer requested refund for unused credits. Processed successfully.",
    ],
    "technical": [
        "Technical support call. Customer experiencing call quality issues. Suggested network troubleshooting.",
        "Customer unable to connect to phone system. Reset credentials and resolved.",
        "Integration issues with Salesforce. Walked through OAuth setup.",
    ],
    "feature_request": [
        "Customer requested call recording feature. Added to product roadmap.",
        "Asked about AI call summaries feature. Explained upcoming Voice Agent product.",
        "Feature request for better analytics dashboard. Noted feedback.",
    ],
    "account_setup": [
        "New customer onboarding call. Set up their first phone numbers and users.",
        "Account upgrade from Basic to Pro plan. Migrated successfully.",
        "Customer adding 10 new team members. Provisioned accounts.",
    ],
    "call_quality": [
        "Customer reporting echo on calls. Diagnosed network latency issue.",
        "Call drop issues during peak hours. Escalated to engineering team.",
        "Audio quality degradation. Recommended bandwidth upgrade.",
    ],
}

AGENTS = ["Sarah", "Mike", "Alex", "Jordan", "Emily", "Chris"]


def generate_call(call_id: int) -> dict:
    """Generate a single fake call record."""
    category = random.choice(list(TOPICS.keys()))
    transcript = random.choice(TOPICS[category])
    
    # Generate random timestamp within last 30 days
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(
        days=days_ago,
        hours=hours_ago,
        minutes=minutes_ago
    )
    
    return {
        "call_id": f"call_{call_id:03d}",
        "from": f"+1-555-{random.randint(1000, 9999)}",
        "to": "+1-555-0200",  # Aircall support line
        "duration": random.randint(60, 600),  # 1-10 minutes
        "timestamp": timestamp.isoformat(),
        "transcript": transcript,
        "tags": category.split("_"),
        "agent": random.choice(AGENTS),
        "sentiment": random.choice(["positive", "neutral", "negative"]),
        "resolved": random.choice([True, False]),
    }


def main():
    """Generate and save fake call data."""
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate 20 calls
    num_calls = 20
    calls = [generate_call(i) for i in range(1, num_calls + 1)]
    
    # Save to JSON
    output_file = data_dir / "calls.json"
    with open(output_file, "w") as f:
        json.dump(calls, f, indent=2)
    
    print(f"✅ Generated {num_calls} fake calls")
    print(f"📁 Saved to {output_file}")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Total calls: {num_calls}")
    print(f"   Agents: {len(AGENTS)}")
    print(f"   Categories: {len(TOPICS)}")
    print(f"   Date range: Last 30 days")


if __name__ == "__main__":
    main()
