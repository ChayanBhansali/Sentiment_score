# import gradio as gr
# import matplotlib.pyplot as plt
# import numpy as np
# import json
# import pandas as pd
# from sqlalchemy.orm import sessionmaker
# from src.db import engine, TextEntry, AnalysisResult

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# db = SessionLocal()

# def fetch_data():
#     query = """
#     SELECT text_entries.id, text_entries.input_text, text_entries.timestamp,
#            analysis_results.emotion_scores, analysis_results.education_scores
#     FROM text_entries
#     JOIN analysis_results ON text_entries.id = analysis_results.text_id
#     """
#     all_entries = db.query(TextEntry).all()

#     processed_data = []
#     for row in all_entries:
#         if row.results:
#             result: AnalysisResult = row.results[0]
#             emotion_scores = result.emotion_scores
#             education_scores = result.education_scores
#             print("emotion_scores", education_scores)
#             print("education_scores", education_scores)
#             scores = {**{e[0]: e[1] for e in emotion_scores},
#                       **{e[0]: e[1] for e in education_scores}}
#             processed_data.append((row.id, row.input_text, row.timestamp, scores))
#     db.close()

#     return processed_data

# def plot_spider_chart(text_id):
#     data = fetch_data()
#     entry = next((d for d in data if d[0] == text_id), None)

#     if not entry:
#         return "No data found", None

#     labels = list(entry[3].keys())
#     scores = list(entry[3].values())

#     num_vars = len(labels)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     scores += scores[:1]
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
#     ax.fill(angles, scores, color='blue', alpha=0.25)
#     ax.plot(angles, scores, color='blue', linewidth=2)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels([])

#     return fig

# def update_table(order):
#     data = fetch_data()
#     df = pd.DataFrame(data, columns=["ID", "Text", "Timestamp", "Scores"])
#     df["Timestamp"] = pd.to_datetime(df["Timestamp"])
#     df = df.sort_values(by="Timestamp", ascending=(order == "Ascending"))
#     return df.drop(columns=["Scores"])

# with gr.Blocks() as app:
#     gr.Markdown("# Text Analysis Dashboard")

#     with gr.Row():
#         text_id_input = gr.Number(label="Enter Text ID for Visualization")
#         plot_output = gr.Plot()
#         text_id_input.change(plot_spider_chart, inputs=text_id_input, outputs=plot_output)

#     with gr.Row():
#         order_dropdown = gr.Dropdown(["Ascending", "Descending"], label="Sort by Timestamp", value="Descending")
#         table_output = gr.Dataframe()
#         order_dropdown.change(update_table, inputs=order_dropdown, outputs=table_output)

#     # table_output = update_table("Descending")

# app.launch(server_name="0.0.0.0", server_port=7860)


import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from src.db import engine, TextEntry, AnalysisResult
from typing import Dict, Tuple
import logging

# Assuming you have these model inference functions defined elsewhere
# from src.models import get_emotion_scores, get_education_scores  # You'll need to implement these

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
# from transformers import pipeline

from transformers import pipeline

def analyze_text(text: str) -> Tuple[Dict, Dict]:
    """
    Analyze input text and return emotion and education scores
    """
    emotion_pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

    # edu_pipe = pipeline("text-classification", model="HuggingFaceFW/fineweb-edu-classifier")
    edu_pipe = pipeline("text-classification", model="HuggingFaceFW/fineweb-edu-classifier")

    emotion_scores = emotion_pipe([text])  # Implement this function
    education_scores = edu_pipe([text]) # Implement this function
    # emotion_scores = get_emotion_scores(text)  # Implement this function
    # education_scores = get_education_scores(text)  # Implement this function
    return emotion_scores, education_scores

def save_to_database(text: str, emotion_scores: Dict, education_scores: Dict) -> int:
    """
    Save the text and analysis results to database
    """
    db = SessionLocal()
    try:
        # Create new text entry
        text_entry = TextEntry(
            input_text=text,
            timestamp=datetime.now()
        )
        db.add(text_entry)
        db.flush()  # Flush to get the ID
        logging.info(emotion_scores)
        logging.info(education_scores)

        # Create analysis result
        analysis_result = AnalysisResult(
            text_id=text_entry.id,
            emotion_scores=[(k, v) for k, v in emotion_scores.items()],
            education_scores=[(k, v) for k, v in education_scores.items()]
        )
        db.add(analysis_result)
        db.commit()
        return text_entry.id
    finally:
        db.close()

def fetch_data():
    """
    Fetch all entries from database
    """
    db = SessionLocal()
    try:
        all_entries = db.query(TextEntry).all()
        processed_data = []
        for row in all_entries:
            if row.results:
                result: AnalysisResult = row.results[0]
                emotion_scores = result.emotion_scores
                education_scores = result.education_scores
                scores = {**{e[0]: e[1] for e in emotion_scores},
                         **{e[0]: e[1] for e in education_scores}}
                processed_data.append((row.id, row.input_text, row.timestamp, scores))
        return processed_data
    finally:
        db.close()

def plot_spider_chart(scores: Dict) -> plt.Figure:
    """
    Create spider chart from scores
    """
    labels = list(scores.keys())
    values = list(scores.values())

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    return fig

def process_text(text: str):
    """
    Process input text and return visualization and updated table
    """
    # Get scores from models
    emotion_scores, education_scores = analyze_text(text)

    # Save to database
    save_to_database(text, emotion_scores, education_scores)

    # Create visualization
    all_scores = {**emotion_scores, **education_scores}
    fig = plot_spider_chart(all_scores)

    # Update table
    data = fetch_data()
    df = pd.DataFrame(data, columns=["ID", "Text", "Timestamp", "Scores"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp", ascending=False)

    return fig, df.drop(columns=["Scores"])

def update_table(order: str):
    """
    Update table based on sort order
    """
    data = fetch_data()
    df = pd.DataFrame(data, columns=["ID", "Text", "Timestamp", "Scores"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp", ascending=(order == "Ascending"))
    return df.drop(columns=["Scores"])

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Real-time Text Analysis Dashboard")

    with gr.Row():
        # Text input for analysis
        text_input = gr.Textbox(label="Enter text to analyze", lines=3)
        analyze_btn = gr.Button("Analyze")

    with gr.Row():
        # Visualization output
        plot_output = gr.Plot()

    with gr.Row():
        # Table controls and output
        order_dropdown = gr.Dropdown(
            ["Ascending", "Descending"],
            label="Sort by Timestamp",
            value="Descending"
        )
        table_output = gr.Dataframe()

    # Set up event handlers
    analyze_btn.click(
        process_text,
        inputs=text_input,
        outputs=[plot_output, table_output]
    )

    order_dropdown.change(
        update_table,
        inputs=order_dropdown,
        outputs=table_output
    )

# Launch the app
app.launch(server_name="0.0.0.0", server_port=7860)