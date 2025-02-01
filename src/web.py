import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import json
import pandas as pd

def fetch_data():
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()
    query = """
    SELECT text_entries.id, text_entries.input_text, text_entries.timestamp, 
           analysis_results.emotion_scores, analysis_results.education_scores 
    FROM text_entries 
    JOIN analysis_results ON text_entries.id = analysis_results.text_id
    """
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    
    processed_data = []
    for row in data:
        emotion_scores = json.loads(row[3])
        education_scores = json.loads(row[4])
        scores = {**{e['label']: e['score'] for e in emotion_scores}, 
                  **{e['label']: e['score'] for e in education_scores}}
        processed_data.append((row[0], row[1], row[2], scores))
    
    return processed_data

def plot_spider_chart(text_id):
    data = fetch_data()
    entry = next((d for d in data if d[0] == text_id), None)
    
    if not entry:
        return "No data found", None
    
    labels = list(entry[3].keys())
    scores = list(entry[3].values())
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.fill(angles, scores, color='blue', alpha=0.25)
    ax.plot(angles, scores, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    
    return fig

def update_table(order):
    data = fetch_data()
    df = pd.DataFrame(data, columns=["ID", "Text", "Timestamp", "Scores"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp", ascending=(order == "Ascending"))
    return df.drop(columns=["Scores"])

with gr.Blocks() as app:
    gr.Markdown("# Text Analysis Dashboard")
    
    with gr.Row():
        text_id_input = gr.Number(label="Enter Text ID for Visualization")
        plot_output = gr.Plot()
        # text_id_input.change(plot_spider_chart, inputs=text_id_input, outputs=plot_output)
    
    with gr.Row():
        order_dropdown = gr.Dropdown(["Ascending", "Descending"], label="Sort by Timestamp", value="Descending")
        table_output = gr.Dataframe()
        # order_dropdown.change(update_table, inputs=order_dropdown, outputs=table_output)
    
    # table_output = update_table("Descending") 

app.launch()
