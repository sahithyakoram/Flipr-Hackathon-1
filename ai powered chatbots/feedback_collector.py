import csv

def collect_feedback(ticket_id, rating, feedback_text):
    try:
        with open("support_tickets.csv", mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            tickets = list(reader)

        updated = False
        for ticket in tickets:
            if ticket['ticket_id'] == ticket_id:
                ticket['rating'] = rating
                ticket['feedback'] = feedback_text
                updated = True
                break

        if not updated:
            print(f"❌ Ticket {ticket_id} not found.")
            return

        with open("support_tickets.csv", mode='w', newline='', encoding='utf-8') as file:
            fieldnames = list(tickets[0].keys()) + ['rating', 'feedback'] if 'rating' not in tickets[0] else tickets[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tickets)

        print(f"📝 Feedback recorded for ticket {ticket_id}!")

    except FileNotFoundError:
        print("⚠️ support_tickets.csv not found.")