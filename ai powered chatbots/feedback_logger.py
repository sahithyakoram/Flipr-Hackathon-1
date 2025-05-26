import csv
from datetime import datetime

def collect_feedback(user_query):
    try:
        rating = int(input("🤔 How helpful was the answer? (1 - Not Helpful, 5 - Very Helpful): "))
        if rating < 1 or rating > 5:
            print("⚠️ Please enter a rating between 1 and 5.")
            return

        comment = input("💬 Any additional feedback? (Optional): ")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        feedback_data = {
            'timestamp': timestamp,
            'query': user_query,
            'rating': rating,
            'comment': comment
        }

        with open("feedback.csv", mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=feedback_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(feedback_data)

        print("🙏 Thank you for your feedback!")
    except ValueError:
        print("⚠️ Invalid rating. Please enter a number between 1 and 5.")