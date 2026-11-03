import pandas as pd
import random

def generate_tweets():
    # Positive and negative keywords for mock data
    pos_words = ["love", "happy", "amazing", "great", "best", "wonderful", "joy", "fantastic", "excellent", "awesome", "beautiful", "sunny", "fun", "cool", "superb"]
    neg_words = ["hate", "sad", "awful", "terrible", "worst", "miserable", "bad", "horrible", "angry", "poor", "boring", "disappointed", "ugly", "failed"]
    
    data = []
    for _ in range(500):
        if _ % 2 == 0:
            # Positive tweet
            words = random.sample(pos_words, 3) 
            text = f"I think this is {words[0]}! Had such a {words[1]} day. #happy @friend"
            sentiment = "Positive"
        else:
            # Negative tweet
            words = random.sample(neg_words, 3)
            text = f"I {words[0]} this. It's so {words[1]} and {words[2]}. My day is ruined."
            sentiment = "Negative"
        
        data.append({"text": text, "sentiment": sentiment})
    
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent
    output_path = data_dir / "tweets.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_tweets()
