import csv

def view_tickets(filter_status=None, filter_priority=None, filter_sentiment=None):
    try:
        with open("support_tickets.csv", mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            tickets = list(reader)

            # Apply filters
            if filter_status:
                tickets = [t for t in tickets if t['status'].lower() == filter_status.lower()]
            if filter_priority:
                tickets = [t for t in tickets if t['priority'].lower() == filter_priority.lower()]
            if filter_sentiment:
                tickets = [t for t in tickets if t['sentiment'].lower() == filter_sentiment.lower()]

            if not tickets:
                print("🙁 No tickets found with those filters.")
                return

            print("\n📋 Filtered Tickets:\n")
            for ticket in tickets:
                print(f"🆔 Ticket ID: {ticket['ticket_id']}")
                print(f"📅 Timestamp: {ticket['timestamp']}")
                print(f"💬 Query: {ticket['user_query']}")
                print(f"🔍 Sentiment: {ticket['sentiment']}")
                print(f"🚦 Priority: {ticket['priority']}")
                print(f"📌 Status: {ticket['status']}")
                print("-" * 50)

    except FileNotFoundError:
        print("⚠️ No support_tickets.csv file found.")

# 🔽 Example usage:
# view_tickets(filter_status="Open", filter_priority="High")
