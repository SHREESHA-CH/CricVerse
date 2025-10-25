# # # Run the app with: uvicorn main:app --reload
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pandas as pd
# from transformers import pipeline
# from fastapi.middleware.cors import CORSMiddleware

# # ---------- Load Models Once (Global) ----------
# MODEL_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/sentence_transformer_model"
# CSV_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/batting_stats_cleaned.csv"
# FAISS_INDEX_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/faiss_index_batting_stats"

# # Load once when app starts
# print("🔹 Loading SentenceTransformer model...")
# model = SentenceTransformer(MODEL_PATH)

# print("🔹 Loading FAISS index...")
# index = faiss.read_index(FAISS_INDEX_PATH)

# print("🔹 Loading dataset...")
# df = pd.read_csv(CSV_PATH)

# print("🔹 Loading QA pipeline...")
# qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# # Conversational rephraser
# rephraser = pipeline("text2text-generation", model="google/flan-t5-small")


# # ---------- FastAPI Setup ----------
# app = FastAPI()

# # Allow your frontend origin
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # for testing; later use ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class InputData(BaseModel):
#     text: str
#     role: str  # "user" or "bot"

# # Global memory for conversation
# conversation_context = {
#     "last_player": None,
#     "last_format": None,
#     "last_data": None
# }

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         query = data.text.lower()
#         global conversation_context

#         # Handle greetings or small talk
#         if any(word in query for word in ["hi", "hello", "hey"]):
#             return {"text": "Hey there! 😊 I'm CricVerse, your cricket assistant. Ask me about any player's stats!", "role": "bot"}
#         if "how are you" in query:
#             return {"text": "I'm doing great and ready to talk cricket! How about you?", "role": "bot"}
#         if "thank" in query:
#             return {"text": "You're most welcome! Always happy to chat cricket 🏏", "role": "bot"}

#         # Determine if player name mentioned
#         player_mentioned = None
#         for player in df["Player"]:
#             if player.lower() in query:
#                 player_mentioned = player
#                 break

#         # If no player mentioned, use memory
#         if not player_mentioned and conversation_context["last_player"]:
#             player_mentioned = conversation_context["last_player"]

#         # If still no player, find best match from FAISS
#         if not player_mentioned:
#             query_embedding = model.encode([query])[0].reshape(1, -1)
#             distances, indices = index.search(query_embedding, 1)
#             closest_row = df.iloc[indices[0][0]]
#         else:
#             # Get player record
#             closest_row = df[df["Player"] == player_mentioned].iloc[0]

#         # Update memory
#         conversation_context["last_player"] = closest_row["Player"]
#         conversation_context["last_data"] = closest_row.to_dict()
#         conversation_context["last_format"] = closest_row.get("Format", "overall")

#         # Detect what user is asking
#         if "run" in query:
#             answer = f"{closest_row['Player']} has scored {closest_row['Runs']} runs in {closest_row['Format']} cricket."
#         elif "match" in query:
#             answer = f"{closest_row['Player']} has played {closest_row['Matches']} matches in {closest_row['Format']} cricket."
#         elif "average" in query:
#             answer = f"{closest_row['Player']}'s batting average in {closest_row['Format']} cricket is {closest_row['Average']}."
#         elif "century" in query or "hundred" in query:
#             answer = f"{closest_row['Player']} has scored {closest_row.get('100s', 'N/A')} centuries in {closest_row['Format']} cricket."
#         elif "fifty" in query:
#             answer = f"{closest_row['Player']} has scored {closest_row.get('50s', 'N/A')} fifties in {closest_row['Format']} cricket."
#         else:
#             # Generic fallback using context
#             answer = f"{closest_row['Player']} ({closest_row['Format']}) - Runs: {closest_row['Runs']}, Matches: {closest_row['Matches']}, Average: {closest_row['Average']}."

#         # Make it conversational
#         prompt = f"Rephrase this like a friendly cricket chatbot: {answer}"
#         rephrased = rephraser(prompt, max_length=50)[0]['generated_text']

#         return {"text": rephrased, "role": "bot"}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# import pandas as pd
# import faiss
# from fastapi.middleware.cors import CORSMiddleware

# # ---------- CONFIG ----------
# MODEL_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/sentence_transformer_model"
# CSV_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/batting_stats_cleaned.csv"
# FAISS_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/faiss_index_batting_stats"

# # ---------- LOAD MODELS ----------
# print("🔹 Loading models and data...")
# model = SentenceTransformer(MODEL_PATH)
# index = faiss.read_index(FAISS_PATH)
# df = pd.read_csv(CSV_PATH)
# rephraser = pipeline("text2text-generation", model="google/flan-t5-small")

# # ---------- APP SETUP ----------
# app = FastAPI(title="CricVerse - AI Cricket Chatbot")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to your frontend URL later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class InputData(BaseModel):
#     text: str
#     role: str

# conversation_context = {
#     "last_player": None,
#     "last_format": None
# }

# # ---------- HELPER FUNCTIONS ----------

# def detect_player(query):
#     """Detect player name in query or return best FAISS match."""
#     global df, model, index

#     for player in df["Player"]:
#         if player.lower() in query:
#             return player

#     # Semantic fallback
#     query_emb = model.encode([query])[0].reshape(1, -1)
#     distances, indices = index.search(query_emb, 1)
#     best_match = df.iloc[indices[0][0]]
#     return best_match["Player"]

# def detect_format(query):
#     formats = ["test", "odi", "t20"]
#     for f in formats:
#         if f in query:
#             return f.upper()
#     return None

# def detect_intent(query):
#     """Simple rule-based intent recognition."""
#     intents = {
#         "runs": ["runs", "scored", "total runs"],
#         "average": ["average", "batting average"],
#         "matches": ["matches", "games", "played"],
#         "centuries": ["centuries", "hundreds", "100s"],
#         "fifties": ["fifties", "50s"],
#         "stats": ["statistics", "details", "record", "performance"]
#     }
#     for key, keywords in intents.items():
#         if any(word in query for word in keywords):
#             return key
#     return "stats"

# # ---------- PREDICT FUNCTION ----------

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         query = data.text.lower()
        
#         # ---------- Detect player ----------
#         player = detect_player(query)
#         fmt = detect_format(query)

#         if not player:
#             return {"text": "I couldn’t find that player. Could you mention the name again?", "role": "bot"}

#         conversation_context["last_player"] = player
#         if fmt:
#             conversation_context["last_format"] = fmt

#         # ---------- Fetch player data ----------
#         player_data = df[df["Player"].str.lower() == player.lower()]
#         if fmt:
#             player_data = player_data[player_data["Format"].str.lower() == fmt.lower()]
#         if player_data.empty:
#             player_data = df[df["Player"].str.lower() == player.lower()].iloc[0]
#         else:
#             player_data = player_data.iloc[0]
            
#         # Before querying player_data
#         if not fmt:
#             # Use last format if available
#             fmt = conversation_context.get("last_format")
#         else:
#             conversation_context["last_format"] = fmt


#         # ---------- Detect intent ----------
#         intent = detect_intent(query)

#         if intent in ["runs", "matches", "average", "centuries", "fifties"]:
#             answer_map = {
#                 "runs": f"{player} has scored {player_data['Runs']} runs in {player_data['Format']} cricket.",
#                 "matches": f"{player} has played {player_data['Matches']} matches in {player_data['Format']} cricket.",
#                 "average": f"{player} has an average of {player_data['Average']} in {player_data['Format']} cricket.",
#                 "centuries": f"{player} has scored {player_data.get('100s', 'N/A')} centuries in {player_data['Format']}.",
#                 "fifties": f"{player} has scored {player_data.get('50s', 'N/A')} fifties in {player_data['Format']}."
#             }
#             return {"text": answer_map[intent], "role": "bot"}
#         if not any(player.lower() in query for player in df["Player"].str.lower()):
#             # Use last player if available
#             player = conversation_context.get("last_player")


#         # ---------- Rephrase for friendly output ----------
#         prompt = f"""
#         You are a friendly cricket expert. Use the exact stats provided and make the answer human-like and conversational.
#         Do NOT make up numbers or add unrelated cricket facts.

#         Example 1:
#         Stat: 7202 runs
#         Answer: "Virat has scored an incredible 7202 runs in Tests! Truly a legend."

#         Example 2:
#         Stat: 242 matches
#         Answer: "He has played 242 matches, showing great consistency."

#         Now, convert this stat into a friendly answer:
#         Stat: {raw_answer}
#         Answer:
#         """
#         rephrased = rephraser(prompt, max_length=100)[0]["generated_text"]

#         return {"text": rephrased, "role": "bot"}

#     except Exception as e:
#         print(f"[MAIN ERROR] {e}")
#         raise HTTPException(status_code=400, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import faiss
from fastapi.middleware.cors import CORSMiddleware

# ---------- CONFIG ----------
MODEL_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/sentence_transformer_model"
CSV_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/batting_stats_cleaned.csv"
FAISS_PATH = "C:/Users/user/Desktop/projects/cricket llm/final project/backend/faiss_index_batting_stats"

# ---------- LOAD MODELS ----------
print("🔹 Loading models and data...")
model = SentenceTransformer(MODEL_PATH)
index = faiss.read_index(FAISS_PATH)
df = pd.read_csv(CSV_PATH)
rephraser = pipeline("text2text-generation", model="google/flan-t5-small")

# ---------- APP SETUP ----------
app = FastAPI(title="CricVerse - AI Cricket Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    text: str
    role: str

conversation_context = {
    "last_player": None,
    "last_format": None
}

# ---------- HELPER FUNCTIONS ----------

def detect_player(query):
    """Detect player name in query or return best FAISS match."""
    global df, model, index

    for player in df["Player"]:
        if player.lower() in query:
            return player

    # Semantic fallback
    query_emb = model.encode([query])[0].reshape(1, -1)
    distances, indices = index.search(query_emb, 1)
    best_match = df.iloc[indices[0][0]]
    return best_match["Player"]

def detect_format(query):
    formats = ["test", "odi", "t20"]
    for f in formats:
        if f in query:
            return f.upper()
    return None

def detect_intent(query):
    """Simple rule-based intent recognition."""
    intents = {
        "runs": ["runs", "scored", "total runs"],
        "average": ["average", "batting average"],
        "matches": ["matches", "games", "played"],
        "centuries": ["centuries", "hundreds", "100s"],
        "fifties": ["fifties", "50s"],
        "stats": ["statistics", "details", "record", "performance"]
    }
    for key, keywords in intents.items():
        if any(word in query for word in keywords):
            return key
    return "stats"

# ---------- PREDICT FUNCTION ----------

@app.post("/predict")
def predict(data: InputData):
    try:
        query = data.text.lower()

        # ---------- Detect player ----------
        player = detect_player(query)
        if not player:
            # fallback to last player in context
            player = conversation_context.get("last_player")
            if not player:
                return {"text": "I couldn’t find that player. Could you mention the name again?", "role": "bot"}

        conversation_context["last_player"] = player

        # ---------- Detect format ----------
        fmt = detect_format(query)
        if not fmt:
            # fallback to last format in context
            fmt = conversation_context.get("last_format")
        else:
            conversation_context["last_format"] = fmt

        # ---------- Fetch player data ----------
        player_data = df[df["Player"].str.lower() == player.lower()]
        if fmt:
            player_data = player_data[player_data["Format"].str.lower() == fmt.lower()]
        if player_data.empty:
            # If no data for this format, fallback to overall
            player_data = df[df["Player"].str.lower() == player.lower()].iloc[0]
        else:
            player_data = player_data.iloc[0]

        # ---------- Detect intent ----------
        intent = detect_intent(query)

        # ---------- Prepare raw answer ----------
        if intent == "runs":
            raw_answer = f"{player_data['Runs']} runs"
        elif intent == "average":
            raw_answer = f"{player_data['Average']} average"
        elif intent == "matches":
            raw_answer = f"{player_data['Matches']} matches"
        elif intent == "centuries":
            raw_answer = f"{player_data.get('100s', 'N/A')} centuries"
        elif intent == "fifties":
            raw_answer = f"{player_data.get('50s', 'N/A')} fifties"
        else:
            # Full stats summary
            raw_answer = (
                f"{player_data['Player']} has played {player_data['Matches']} matches, "
                f"scored {player_data['Runs']} runs, averaging {player_data['Average']}. "
                f"They have {player_data.get('100s', 'N/A')} centuries and {player_data.get('50s', 'N/A')} fifties."
            )

        # ---------- Rephrase for friendly output using few-shot ----------
        prompt = f"""
You are a friendly cricket expert. Use the exact stats provided and make the answer human-like and conversational.
Do NOT make up numbers or add unrelated cricket facts.

Example 1:
Stat: 7202 runs
Answer: "Virat has scored an incredible 7202 runs in Tests! Truly a legend."

Example 2:
Stat: 242 matches
Answer: "He has played 242 matches, showing great consistency."

Now, convert this stat into a friendly answer:
Stat: {raw_answer}
Answer:
"""
        rephrased = rephraser(prompt, max_length=100)[0]["generated_text"]

        return {"text": rephrased, "role": "bot"}

    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        raise HTTPException(status_code=400, detail=str(e))
