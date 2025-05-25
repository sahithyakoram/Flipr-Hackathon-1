import pandas as pd
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt

def analyze_tickets(file_path="support_tickets.csv"):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print("📭 No tickets to analyze.")
            return

        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df['timestamp'].str.startswith(today)]

        if today_df.empty:
            print("📭 No tickets found for today.")
            return

        total = len(today_df)
        open_tickets = (today_df['status'] == 'Open').sum()
        closed_tickets = (today_df['status'] == 'Closed').sum()
        high_priority = (today_df['priority'] == 'High').sum()

        keywords = Counter()
        for query in today_df['user_query']:
            for word in query.lower().split():
                if len(word) > 3 and word.isalpha():
                    keywords[word] += 1

        top_keywords = keywords.most_common(5)

        print("\n📈 Daily Support Summary:")
        print(f"🧾 Date: {today}")
        print(f"📌 Total tickets: {total}")
        print(f"🟡 Open: {open_tickets} | ✅ Closed: {closed_tickets}")
        print(f"🔺 High priority: {high_priority}")
        print("🔥 Top Issues (Keywords):", [kw[0] for kw in top_keywords])

        # 🔵 Bar Chart: Open vs Closed
        plt.figure(figsize=(6, 4))
        plt.bar(['Open', 'Closed'], [open_tickets, closed_tickets], color=['orange', 'green'])
        plt.title('Ticket Status Breakdown')
        plt.ylabel('Count')
        plt.savefig("status_breakdown.png")
        plt.close()

        # 🔴 Pie Chart: Priority
        priorities = today_df['priority'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(priorities, labels=priorities.index, autopct='%1.1f%%', colors=['red', 'yellow', 'lightgreen'])
        plt.title('Priority Distribution')
        plt.savefig("priority_pie_chart.png")
        plt.close()

        # 🟣 Bar Chart: Top 5 keywords
        if top_keywords:
            keywords, counts = zip(*top_keywords)
            plt.figure(figsize=(6, 4))
            plt.bar(keywords, counts, color='skyblue')
            plt.title('Top 5 Keywords Today')
            plt.ylabel('Frequency')
            plt.savefig("top_keywords.png")
            plt.close()

        print("🖼️ Graphs saved: status_breakdown.png, priority_pie_chart.png, top_keywords.png")

    except FileNotFoundError:
        print("❌ No support_tickets.csv found.")