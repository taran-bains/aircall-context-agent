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
        # Distractor: Positive billing mention (vector search will find this, reranker should filter for "issues")
        "Customer called to praise the clear billing format. Very happy with the new invoice design.", 
    ],
    "technical": [
        "Technical support call. Customer experiencing call quality issues. Suggested network troubleshooting.",
        "Customer unable to connect to phone system. Reset credentials and resolved.",
        "Integration issues with Salesforce. Walked through OAuth setup.",
        # Distractor: Technical discussion but not a problem
        "Technical discussion about API capabilities. Customer is building a custom integration.",
    ],
    "call_quality": [
        "Customer reporting echo on calls. Diagnosed network latency issue.",
        "Call drop issues during peak hours. Escalated to engineering team.",
        "Audio quality degradation. Recommended bandwidth upgrade.",
        # Distractor: Feature discussion, not a quality problem
        "Sales call demonstrating high call quality features. Customer was impressed by the audio clarity.",
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
    
    # FORCE DISTRACTORS: Overwrite first 3 calls to ensure we have tricky cases
    
    # 1. Distractor: "Frustration" keyword but POSITIVE context
    calls[0] = {
        "call_id": "call_001_pos_frustration",
        "from": "+1-555-0001",
        "to": "+1-555-0200",
        "duration": 120,
        "timestamp": datetime.now().isoformat(),
        "transcript": "Customer expressed frustration that they couldn't upgrade fast enough because they love the service so much. Wants Enterprise plan immediately.",
        "tags": ["upgrade", "sales"],
        "agent": "Sarah",
        "sentiment": "positive",
        "resolved": True,
    }
    
    # 2. Distractor: "Angry" keyword but irrelevant context (movie discussion)
    calls[1] = {
        "call_id": "call_002_irrelevant",
        "from": "+1-555-0002",
        "to": "+1-555-0200",
        "duration": 180,
        "timestamp": datetime.now().isoformat(),
        "transcript": "Agent and customer chatted about an angry movie character while waiting for system to load. Friendly conversation.",
        "tags": ["chitchat"],
        "agent": "Mike",
        "sentiment": "positive",
        "resolved": True,
    }
    
    # 3. Real Issue: Actual frustration/anger
    calls[2] = {
        "call_id": "call_003_real_anger",
        "from": "+1-555-0003",
        "to": "+1-555-0200",
        "duration": 240,
        "timestamp": datetime.now().isoformat(),
        "transcript": "Customer is furious about recurring downtime. Threatened to cancel contract if not fixed today.",
        "tags": ["technical", "churn_risk"],
        "agent": "Chris",
        "sentiment": "negative",
        "resolved": False,
    }
    
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
