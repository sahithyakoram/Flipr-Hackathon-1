import csv

def update_ticket_status(ticket_id, new_status):
    try:
        with open("support_tickets.csv", mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            tickets = list(reader)

        updated = False
        for ticket in tickets:
            if ticket['ticket_id'] == ticket_id:
                ticket['status'] = new_status
                updated = True
                break

        if not updated:
            print(f"❌ Ticket {ticket_id} not found.")
            return

        with open("support_tickets.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=tickets[0].keys())
            writer.writeheader()
            writer.writerows(tickets)

        print(f"✅ Ticket {ticket_id} updated to status '{new_status}'.")

    except FileNotFoundError:
        print("⚠️ support_tickets.csv not found.")