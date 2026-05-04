from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
from collections import Counter

app = FastAPI(title="Roman Urdu Sentiment API", version="1.0")

print("Loading model...")
classifier = pipeline("text-classification", model="./final_model_v6", tokenizer="./final_model_v6")
print("Model loaded.")

label_map = {"LABEL_2": "positive", "LABEL_1": "neutral", "LABEL_0": "negative"}

def post_process(text: str, sentiment: str, confidence: float):
    t = text.lower()
    if "kamaal kar diya" in t or "kya baat hai kamaal" in t:
        return "positive", confidence
    if "nirash kiya" in t or "bohat nirash" in t:
        return "negative", confidence
    if "is se acha to main khud kar leta" in t:
        return "negative", confidence
    if "bakwas recommendation" in t:
        return "negative", confidence
    if "bohat bura laga" in t or "bura laga jaankar" in t:
        return "negative", confidence
    if "behtar ho sakta tha" in t and ("lekin theek hai" in t or "magar acceptable" in t):
        return "neutral", confidence
    if "theek hai, lekin behtar ho sakta tha" in t:
        return "neutral", confidence
    return sentiment, confidence

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    result = classifier(request.text)[0]
    raw_sentiment = label_map[result['label']]
    confidence = result['score']
    final_sentiment, final_confidence = post_process(request.text, raw_sentiment, confidence)
    return SentimentResponse(
        text=request.text,
        sentiment=final_sentiment,
        confidence=round(final_confidence, 4)
    )

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Only CSV or TXT files allowed")
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
        if 'text' in df.columns:
            sentences = df['text'].tolist()
        else:
            sentences = df.iloc[:, 0].tolist()
    else:
        text_data = contents.decode('utf-8')
        sentences = [line.strip() for line in text_data.splitlines() if line.strip()]

    results = []
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for sent in sentences:
        if not sent:
            continue
        result = classifier(sent)[0]
        raw_sentiment = label_map[result['label']]
        confidence = result['score']
        final_sentiment, final_confidence = post_process(sent, raw_sentiment, confidence)
        results.append({
            "text": sent,
            "sentiment": final_sentiment,
            "confidence": round(final_confidence, 4)
        })
        sentiment_counts[final_sentiment] += 1

    # Generate bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    sentiments = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    colors = ['#28a745', '#dc3545', '#ffc107']
    bars = ax.bar(sentiments, counts, color=colors)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Number of Sentences')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')
    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return {
        "filename": file.filename,
        "total": len(results),
        "results": results,
        "chart": chart_base64,
        "counts": sentiment_counts
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Roman Urdu Sentiment Analyzer</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                width: 100%;
                background: rgba(255,255,255,0.95);
                border-radius: 30px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 { text-align: center; color: #333; margin-bottom: 10px; font-size: 2.5rem; }
            .sub { text-align: center; color: #666; margin-bottom: 40px; }
            .row { display: flex; gap: 30px; flex-wrap: wrap; }
            .card {
                flex: 1;
                min-width: 280px;
                background: #f9f9ff;
                border-radius: 20px;
                padding: 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .card:hover { transform: translateY(-5px); }
            .card h2 { color: #4a4a8a; margin-bottom: 20px; border-left: 5px solid #764ba2; padding-left: 15px; }
            input, textarea {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #ddd;
                border-radius: 12px;
                font-size: 1rem;
            }
            input:focus, textarea:focus { border-color: #764ba2; outline: none; }
            button {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 40px;
                font-size: 1rem;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
            }
            button:hover { transform: scale(1.02); }
            .file-label {
                background: #e9ecef;
                padding: 12px;
                border-radius: 12px;
                text-align: center;
                cursor: pointer;
                display: block;
                margin: 10px 0;
                border: 2px dashed #764ba2;
            }
            input[type="file"] { display: none; }
            .result-area {
                margin-top: 30px;
                background: #f1f3f9;
                border-radius: 20px;
                padding: 20px;
                max-height: 400px;
                overflow-y: auto;
            }
            .result-item {
                background: white;
                border-radius: 12px;
                padding: 12px;
                margin-bottom: 10px;
                border-left: 5px solid;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .positive { border-left-color: #28a745; }
            .negative { border-left-color: #dc3545; }
            .neutral { border-left-color: #ffc107; }
            .sentiment-badge {
                font-weight: bold;
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                margin-left: 10px;
            }
            .badge-pos { background: #28a74520; color: #155724; }
            .badge-neg { background: #dc354520; color: #721c24; }
            .badge-neu { background: #ffc10720; color: #856404; }
            .chart-container { text-align: center; margin-top: 20px; }
            img { max-width: 100%; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            hr { margin: 20px 0; }
            @media (max-width: 768px) { .container { padding: 20px; } .row { flex-direction: column; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✨ Roman Urdu Sentiment Analyzer ✨</h1>
            <div class="sub">Code-Mixed (Urdu + English) | AI Powered | Fast & Accurate</div>
            <div class="row">
                <div class="card">
                    <h2>📝 Single Sentence</h2>
                    <input type="text" id="singleText" placeholder="Type your sentence here... e.g., yeh movie bohat achi thi">
                    <button id="singleBtn">🔍 Analyze Sentiment</button>
                    <div id="singleResult" class="result-area" style="display:none;"></div>
                </div>
                <div class="card">
                    <h2>📂 Bulk Upload (CSV/TXT)</h2>
                    <label class="file-label" for="fileInput">📎 Choose File (CSV or TXT)</label>
                    <input type="file" id="fileInput" accept=".csv,.txt">
                    <button id="fileBtn">🚀 Upload & Analyze All</button>
                    <div id="fileResult" class="result-area" style="display:none;"></div>
                    <div id="chartDiv" class="chart-container" style="display:none;"></div>
                </div>
            </div>
        </div>
        <script>
            const singleBtn = document.getElementById('singleBtn');
            const singleText = document.getElementById('singleText');
            const singleResultDiv = document.getElementById('singleResult');
            const fileBtn = document.getElementById('fileBtn');
            const fileInput = document.getElementById('fileInput');
            const fileResultDiv = document.getElementById('fileResult');
            const chartDiv = document.getElementById('chartDiv');

            singleBtn.addEventListener('click', async () => {
                const text = singleText.value.trim();
                if (!text) { alert('Please enter a sentence'); return; }
                singleResultDiv.style.display = 'block';
                singleResultDiv.innerHTML = '<div style="text-align:center">⏳ Analyzing...</div>';
                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await res.json();
                    const badgeClass = data.sentiment === 'positive' ? 'badge-pos' : (data.sentiment === 'negative' ? 'badge-neg' : 'badge-neu');
                    const borderClass = data.sentiment;
                    singleResultDiv.innerHTML = `
                        <div class="result-item ${borderClass}">
                            <strong>📌 Text:</strong> ${escapeHtml(data.text)}<br>
                            <strong>🎯 Sentiment:</strong> 
                            <span class="sentiment-badge ${badgeClass}">${data.sentiment.toUpperCase()}</span>
                            <br>
                            <strong>📊 Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
                        </div>
                    `;
                } catch (err) {
                    singleResultDiv.innerHTML = `<div class="result-item negative">❌ Error: ${err.message}</div>`;
                }
            });

            fileBtn.addEventListener('click', async () => {
                if (!fileInput.files.length) { alert('Please select a file first'); return; }
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                fileResultDiv.style.display = 'block';
                chartDiv.style.display = 'none';
                fileResultDiv.innerHTML = '<div style="text-align:center">⏳ Processing file... (may take a few seconds)</div>';
                try {
                    const res = await fetch('/predict_file', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    let html = `<div><strong>📄 File:</strong> ${data.filename} | <strong>📝 Total:</strong> ${data.total} sentences</div><hr>`;
                    data.results.forEach(item => {
                        const badgeClass = item.sentiment === 'positive' ? 'badge-pos' : (item.sentiment === 'negative' ? 'badge-neg' : 'badge-neu');
                        const borderClass = item.sentiment;
                        html += `
                            <div class="result-item ${borderClass}">
                                <strong>Text:</strong> ${escapeHtml(item.text)}<br>
                                <strong>Sentiment:</strong> <span class="sentiment-badge ${badgeClass}">${item.sentiment.toUpperCase()}</span>
                                &nbsp; <strong>Confidence:</strong> ${(item.confidence * 100).toFixed(2)}%
                            </div>
                        `;
                    });
                    fileResultDiv.innerHTML = html;
                    // Show chart if present
                    if (data.chart) {
                        chartDiv.innerHTML = `<h3>📊 Sentiment Distribution</h3><img src="data:image/png;base64,${data.chart}" alt="Sentiment Chart">`;
                        chartDiv.style.display = 'block';
                    }
                } catch (err) {
                    fileResultDiv.innerHTML = `<div class="result-item negative">❌ Error: ${err.message}</div>`;
                }
            });

            function escapeHtml(str) {
                return str.replace(/[&<>]/g, function(m) {
                    if (m === '&') return '&amp;';
                    if (m === '<') return '&lt;';
                    if (m === '>') return '&gt;';
                    return m;
                });
            }
        </script>
    </body>
    </html>
    """