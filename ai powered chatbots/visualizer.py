import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("ai powered chatbots/support_tickets.csv", parse_dates=["timestamp"])

# Plot 1: Tickets over time
df["date"] = df["timestamp"].dt.date
tickets_per_day = df.groupby("date").size()
plt.figure(figsize=(8, 4))
tickets_per_day.plot(kind="bar", title="Tickets Created Per Day")
plt.xlabel("Date")
plt.ylabel("Number of Tickets")
plt.tight_layout()
plt.show()

# Plot 2: Priority distribution
priority_counts = df["priority"].value_counts()
priority_counts.plot(kind="pie", autopct="%1.1f%%", title="Ticket Priority Distribution", ylabel="")
plt.tight_layout()
plt.show()

# Plot 3: Sentiment breakdown
sentiment_counts = df["sentiment"].value_counts()
sentiment_counts.plot(kind="bar", title="User Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()