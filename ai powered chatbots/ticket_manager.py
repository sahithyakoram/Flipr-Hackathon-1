import csv
import uuid
from datetime import datetime

def create_support_ticket(user_query, sentiment='neutral', priority='Normal'):
    ticket_id = f"TCK{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ticket_data = {
        'ticket_id': ticket_id,
        'timestamp': timestamp,
        'user_query': user_query,
        'sentiment': sentiment,
        'priority': priority,
        'status': 'Open'
    }

    with open("support_tickets.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=ticket_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(ticket_data)

    print(f"\n🚨 No confident answer found. Ticket {ticket_id} created and assigned to the support team.")