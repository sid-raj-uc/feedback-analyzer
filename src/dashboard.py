"""
Interactive Dashboard Server
==============================
Flask-based dashboard for exploring NLP analysis results.
Serves an interactive single-page application with:
  - Sentiment distribution visualizations
  - Topic cluster exploration
  - Pain point ranking table
  - Category drilldowns
  - Search & filter capabilities

Usage:
    python src/dashboard.py
    → Opens at http://127.0.0.1:5000
"""

import json
import logging
from pathlib import Path

import yaml
import pandas as pd
from flask import Flask, render_template_string, jsonify, send_from_directory

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Data Loaders ───────────────────────────────────────────────────────────────

def load_data():
    """Load all analysis results."""
    data = {}

    # Sentiment / topic results
    topic_path = ROOT_DIR / config["topic_model"]["output_path"]
    sentiment_path = ROOT_DIR / config["sentiment"]["output_path"]

    if topic_path.exists():
        data["reviews"] = pd.read_csv(topic_path)
    elif sentiment_path.exists():
        data["reviews"] = pd.read_csv(sentiment_path)
    else:
        data["reviews"] = pd.DataFrame()

    # Pain points
    pain_path = ROOT_DIR / config["pain_points"]["output_path"]
    if pain_path.exists():
        with open(pain_path) as f:
            data["pain_points"] = json.load(f)
    else:
        data["pain_points"] = {}

    # Topic summary
    topic_summary_path = ROOT_DIR / "outputs/topic_summary.json"
    if topic_summary_path.exists():
        with open(topic_summary_path) as f:
            data["topic_summary"] = json.load(f)
    else:
        data["topic_summary"] = {}

    return data


# ── HTML Template ──────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Feedback Analyzer — Dashboard</title>
    <meta name="description" content="Interactive NLP dashboard for analyzing customer feedback sentiment and topic clusters.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
    <style>
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: #1a1f35;
            --bg-card-hover: #222845;
            --text-primary: #f0f2f5;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-pink: #ec4899;
            --border: #2d3548;
            --border-glow: rgba(59, 130, 246, 0.3);
            --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            --radius: 16px;
            --radius-sm: 10px;
            --glass: rgba(255, 255, 255, 0.03);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated gradient background */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(59, 130, 246, 0.07) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.07) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 60%);
            z-index: -1;
            animation: bgPulse 15s ease-in-out infinite;
        }

        @keyframes bgPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Header */
        .header {
            padding: 32px 48px;
            border-bottom: 1px solid var(--border);
            background: rgba(17, 24, 39, 0.8);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1440px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
        }

        .header-badge {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-nlp {
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            animation: livePulse 2s ease-in-out infinite;
        }

        @keyframes livePulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            50% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        }

        /* Main content */
        .main {
            max-width: 1440px;
            margin: 0 auto;
            padding: 32px 48px;
        }

        /* KPI Cards */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }

        .kpi-card {
            background: var(--bg-card);
            border-radius: var(--radius);
            padding: 24px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .kpi-card:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-glow);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            border-radius: var(--radius) var(--radius) 0 0;
        }

        .kpi-card:nth-child(1)::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
        .kpi-card:nth-child(2)::before { background: linear-gradient(90deg, var(--accent-green), #34d399); }
        .kpi-card:nth-child(3)::before { background: linear-gradient(90deg, var(--accent-red), var(--accent-pink)); }
        .kpi-card:nth-child(4)::before { background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue)); }
        .kpi-card:nth-child(5)::before { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
        .kpi-card:nth-child(6)::before { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)); }

        .kpi-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -1px;
            line-height: 1;
            margin-bottom: 6px;
        }

        .kpi-sub {
            font-size: 13px;
            color: var(--text-secondary);
        }

        /* Section cards */
        .section {
            background: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            margin-bottom: 24px;
            overflow: hidden;
            transition: border-color 0.3s ease;
        }

        .section:hover {
            border-color: rgba(139, 92, 246, 0.2);
        }

        .section-header {
            padding: 24px 28px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .section-title {
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title .icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .section-body {
            padding: 28px;
        }

        /* Charts grid */
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .chart-container {
            background: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            padding: 24px;
            transition: all 0.3s ease;
        }

        .chart-container:hover {
            border-color: var(--border-glow);
        }

        .chart-title {
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        canvas {
            max-width: 100%;
        }

        /* Pain points table */
        .pain-table {
            width: 100%;
            border-collapse: collapse;
        }

        .pain-table th {
            text-align: left;
            padding: 12px 16px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }

        .pain-table td {
            padding: 16px;
            border-bottom: 1px solid rgba(45, 53, 72, 0.5);
            vertical-align: top;
        }

        .pain-table tr {
            transition: background 0.2s ease;
        }

        .pain-table tr:hover {
            background: rgba(59, 130, 246, 0.05);
        }

        .pain-rank {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
        }

        .pain-rank-1 { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .pain-rank-2 { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
        .pain-rank-3 { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
        .pain-rank-default { background: var(--bg-secondary); color: var(--text-secondary); border: 1px solid var(--border); }

        .keyword-tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            margin: 2px 3px;
            background: rgba(139, 92, 246, 0.12);
            color: var(--accent-purple);
            border: 1px solid rgba(139, 92, 246, 0.2);
        }

        .score-bar {
            height: 6px;
            border-radius: 3px;
            background: var(--bg-secondary);
            overflow: hidden;
            margin-top: 4px;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--accent-red), var(--accent-amber));
            transition: width 0.8s ease;
        }

        .severity-badge {
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
        }

        .severity-high {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
        }

        .severity-medium {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
        }

        .severity-low {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
        }

        /* Quote */
        .quote {
            font-style: italic;
            font-size: 13px;
            color: var(--text-secondary);
            padding: 8px 12px;
            border-left: 3px solid var(--accent-purple);
            margin: 4px 0;
            background: rgba(139, 92, 246, 0.05);
            border-radius: 0 6px 6px 0;
            line-height: 1.5;
        }

        /* Topics grid */
        .topics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 16px;
        }

        .topic-card {
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
            padding: 20px;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }

        .topic-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(6, 182, 212, 0.1);
        }

        .topic-id {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-cyan);
            margin-bottom: 8px;
        }

        .topic-count {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .topic-keywords {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }

        /* Responsive image grid for plots */
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .plot-img {
            width: 100%;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            transition: transform 0.3s ease;
        }

        .plot-img:hover {
            transform: scale(1.02);
        }

        /* Tabs */
        .tab-bar {
            display: flex;
            gap: 4px;
            padding: 4px;
            background: var(--bg-secondary);
            border-radius: 12px;
            margin-bottom: 24px;
        }

        .tab {
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            color: var(--text-muted);
            border: none;
            background: transparent;
        }

        .tab:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab.active {
            background: var(--accent-blue);
            color: white;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Category bars */
        .category-row {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 0;
            border-bottom: 1px solid rgba(45, 53, 72, 0.3);
        }

        .category-name {
            width: 160px;
            font-size: 13px;
            font-weight: 500;
            flex-shrink: 0;
        }

        .category-bar-track {
            flex: 1;
            height: 28px;
            background: var(--bg-secondary);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
        }

        .category-bar-seg {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            transition: width 0.8s ease;
        }

        .seg-pos { background: rgba(16, 185, 129, 0.6); color: #d1fae5; }
        .seg-neu { background: rgba(245, 158, 11, 0.4); color: #fef3c7; }
        .seg-neg { background: rgba(239, 68, 68, 0.5); color: #fee2e2; }

        /* Footer */
        .footer {
            text-align: center;
            padding: 32px;
            color: var(--text-muted);
            font-size: 13px;
            border-top: 1px solid var(--border);
            margin-top: 48px;
        }

        /* Loading spinner */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 60px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header { padding: 16px 20px; }
            .main { padding: 16px 20px; }
            .charts-grid { grid-template-columns: 1fr; }
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
            .topics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1>🔍 Customer Feedback Analyzer</h1>
            <div class="header-badge">
                <span class="badge badge-nlp">NLP Pipeline</span>
                <span class="badge badge-live">● Live</span>
            </div>
        </div>
    </header>

    <main class="main" id="app">
        <div class="loading" id="loading">
            <div class="spinner"></div>
        </div>
    </main>

    <script>
        async function loadDashboard() {
            try {
                const resp = await fetch('/api/data');
                const data = await resp.json();
                renderDashboard(data);
            } catch (err) {
                document.getElementById('app').innerHTML =
                    '<p style="color: var(--accent-red); text-align:center; padding:60px;">Error loading data. Make sure the pipeline has been run.</p>';
            }
        }

        function renderDashboard(data) {
            const app = document.getElementById('app');
            const stats = data.stats;
            const painPoints = data.pain_points?.pain_points || [];
            const topics = data.topics || {};
            const categoryStats = data.category_stats || {};

            let html = '';

            // ── KPI Cards ───────────────────────────────────────
            html += `
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Reviews</div>
                    <div class="kpi-value" style="color: var(--accent-blue)">${stats.total_reviews.toLocaleString()}</div>
                    <div class="kpi-sub">${stats.categories} categories analyzed</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Positive</div>
                    <div class="kpi-value" style="color: var(--accent-green)">${stats.positive_pct}%</div>
                    <div class="kpi-sub">${stats.positive_count.toLocaleString()} reviews</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Negative</div>
                    <div class="kpi-value" style="color: var(--accent-red)">${stats.negative_pct}%</div>
                    <div class="kpi-sub">${stats.negative_count.toLocaleString()} reviews</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Topics Found</div>
                    <div class="kpi-value" style="color: var(--accent-purple)">${stats.topics_count}</div>
                    <div class="kpi-sub">${stats.outlier_pct}% outlier rate</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg. Sentiment</div>
                    <div class="kpi-value" style="color: ${stats.avg_compound >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${stats.avg_compound >= 0 ? '+' : ''}${stats.avg_compound}</div>
                    <div class="kpi-sub">Compound VADER score</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Pain Points</div>
                    <div class="kpi-value" style="color: var(--accent-amber)">${painPoints.length}</div>
                    <div class="kpi-sub">Actionable insights surfaced</div>
                </div>
            </div>`;

            // ── Tab Navigation ──────────────────────────────────
            html += `
            <div class="tab-bar">
                <button class="tab active" onclick="switchTab('sentiment')">📊 Sentiment</button>
                <button class="tab" onclick="switchTab('painpoints')">🔴 Pain Points</button>
                <button class="tab" onclick="switchTab('topics')">🧩 Topics</button>
                <button class="tab" onclick="switchTab('categories')">📁 Categories</button>
                <button class="tab" onclick="switchTab('plots')">📈 Visualizations</button>
            </div>`;

            // ── Tab: Sentiment ──────────────────────────────────
            html += `<div class="tab-content active" id="tab-sentiment">
                <div class="charts-grid">
                    <div class="chart-container">
                        <div class="chart-title">Sentiment Distribution</div>
                        <canvas id="sentimentPie" height="280"></canvas>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">Compound Score Histogram</div>
                        <canvas id="compoundHist" height="280"></canvas>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">Sentiment by Star Rating</div>
                        <canvas id="starSentiment" height="280"></canvas>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">Sentiment by Category</div>
                        <canvas id="categorySentiment" height="280"></canvas>
                    </div>
                </div>
            </div>`;

            // ── Tab: Pain Points ────────────────────────────────
            html += `<div class="tab-content" id="tab-painpoints">
                <div class="section">
                    <div class="section-header">
                        <div class="section-title">
                            <span class="icon" style="background: rgba(239,68,68,0.15);">🔴</span>
                            Top Actionable Pain Points
                        </div>
                    </div>
                    <div class="section-body">
                        <table class="pain-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Keywords</th>
                                    <th>Frequency</th>
                                    <th>Severity</th>
                                    <th>Score</th>
                                    <th>Avg Stars</th>
                                    <th>Sample Feedback</th>
                                </tr>
                            </thead>
                            <tbody>`;

            painPoints.forEach(pp => {
                const rankClass = pp.rank <= 3 ? `pain-rank-${pp.rank}` : 'pain-rank-default';
                const sevClass = pp.severity >= 0.7 ? 'severity-high' : pp.severity >= 0.4 ? 'severity-medium' : 'severity-low';
                const keywords = (pp.keywords || []).slice(0, 5).map(k => `<span class="keyword-tag">${k}</span>`).join('');
                const quote = pp.representative_quotes?.[0] || 'N/A';
                const truncQuote = quote.length > 120 ? quote.slice(0, 120) + '…' : quote;

                html += `
                    <tr>
                        <td><div class="pain-rank ${rankClass}">${pp.rank}</div></td>
                        <td>${keywords}</td>
                        <td>
                            <strong>${pp.frequency}</strong>
                            <div class="score-bar"><div class="score-bar-fill" style="width: ${Math.min(pp.frequency / (painPoints[0]?.frequency || 1) * 100, 100)}%"></div></div>
                        </td>
                        <td><span class="severity-badge ${sevClass}">${pp.severity.toFixed(3)}</span></td>
                        <td><strong>${pp.composite_score.toFixed(3)}</strong></td>
                        <td>★ ${pp.avg_star_rating.toFixed(1)}</td>
                        <td><div class="quote">"${truncQuote}"</div></td>
                    </tr>`;
            });

            html += `</tbody></table></div></div></div>`;

            // ── Tab: Topics ─────────────────────────────────────
            html += `<div class="tab-content" id="tab-topics">
                <div class="topics-grid">`;

            Object.entries(topics).forEach(([tid, topic]) => {
                const keywords = (topic.keywords || []).slice(0, 6).map(k => `<span class="keyword-tag">${k.word}</span>`).join('');
                html += `
                    <div class="topic-card">
                        <div class="topic-id">Topic ${tid}</div>
                        <div class="topic-count">${topic.count.toLocaleString()} reviews</div>
                        <div class="topic-keywords">${keywords}</div>
                    </div>`;
            });

            html += `</div></div>`;

            // ── Tab: Categories ─────────────────────────────────
            html += `<div class="tab-content" id="tab-categories">
                <div class="section">
                    <div class="section-header">
                        <div class="section-title">
                            <span class="icon" style="background: rgba(59,130,246,0.15);">📁</span>
                            Sentiment Distribution by Category
                        </div>
                    </div>
                    <div class="section-body">`;

            Object.entries(categoryStats).forEach(([cat, cs]) => {
                const total = cs.positive + cs.negative + cs.neutral;
                const posPct = (cs.positive / total * 100).toFixed(1);
                const neuPct = (cs.neutral / total * 100).toFixed(1);
                const negPct = (cs.negative / total * 100).toFixed(1);

                html += `
                    <div class="category-row">
                        <div class="category-name">${cat}</div>
                        <div class="category-bar-track">
                            <div class="category-bar-seg seg-pos" style="width: ${posPct}%">${posPct}%</div>
                            <div class="category-bar-seg seg-neu" style="width: ${neuPct}%">${neuPct > 8 ? neuPct + '%' : ''}</div>
                            <div class="category-bar-seg seg-neg" style="width: ${negPct}%">${negPct}%</div>
                        </div>
                    </div>`;
            });

            html += `</div></div></div>`;

            // ── Tab: Visualizations (saved plots) ───────────────
            html += `<div class="tab-content" id="tab-plots">
                <div class="plot-grid">`;

            const plots = [
                'sentiment_distribution.png',
                'sentiment_by_category.png',
                'sentiment_vs_stars.png',
                'topic_distribution.png',
                'sentiment_heatmap.png',
                'negative_wordcloud.png',
                'positive_wordcloud.png',
                'length_vs_sentiment.png',
            ];

            plots.forEach(p => {
                html += `<img class="plot-img" src="/plots/${p}" alt="${p}" onerror="this.style.display='none'">`;
            });

            html += `</div></div>`;

            // ── Footer ──────────────────────────────────────────
            html += `
            <div class="footer">
                NLP Customer Feedback Analyzer &middot; VADER Sentiment + BERTopic Clustering &middot;
                ${stats.total_reviews.toLocaleString()} reviews analyzed
            </div>`;

            app.innerHTML = html;

            // ── Render Charts ───────────────────────────────────
            renderCharts(data);
        }

        function renderCharts(data) {
            const stats = data.stats;
            const categoryStats = data.category_stats || {};

            // Sentiment pie
            new Chart(document.getElementById('sentimentPie'), {
                type: 'doughnut',
                data: {
                    labels: ['Positive', 'Negative', 'Neutral'],
                    datasets: [{
                        data: [stats.positive_count, stats.negative_count, stats.neutral_count],
                        backgroundColor: ['#10b981', '#ef4444', '#f59e0b'],
                        borderWidth: 0,
                        hoverOffset: 8,
                    }]
                },
                options: {
                    cutout: '60%',
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#9ca3af', padding: 16 } }
                    },
                    animation: { animateRotate: true, duration: 1200 }
                }
            });

            // Compound histogram
            const bins = data.histogram_bins || [];
            const counts = data.histogram_counts || [];
            new Chart(document.getElementById('compoundHist'), {
                type: 'bar',
                data: {
                    labels: bins.map(b => b.toFixed(2)),
                    datasets: [{
                        data: counts,
                        backgroundColor: bins.map(b => b < -0.05 ? '#ef4444aa' : b > 0.05 ? '#10b981aa' : '#f59e0baa'),
                        borderWidth: 0,
                        borderRadius: 2,
                    }]
                },
                options: {
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: '#6b7280', maxTicksLimit: 10 }, grid: { color: '#1f2937' } },
                        y: { ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } }
                    }
                }
            });

            // Star sentiment
            const starData = data.star_sentiment || {};
            new Chart(document.getElementById('starSentiment'), {
                type: 'bar',
                data: {
                    labels: Object.keys(starData),
                    datasets: [{
                        label: 'Mean Compound',
                        data: Object.values(starData),
                        backgroundColor: Object.values(starData).map(v =>
                            v > 0 ? '#10b981aa' : '#ef4444aa'
                        ),
                        borderRadius: 6,
                    }]
                },
                options: {
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { title: { display: true, text: 'Star Rating', color: '#9ca3af' }, ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } },
                        y: { title: { display: true, text: 'Avg Compound Score', color: '#9ca3af' }, ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } }
                    }
                }
            });

            // Category sentiment
            const catLabels = Object.keys(categoryStats);
            const catAvgs = catLabels.map(c => {
                const cs = categoryStats[c];
                const total = cs.positive + cs.negative + cs.neutral;
                return ((cs.positive - cs.negative) / total).toFixed(3);
            });

            new Chart(document.getElementById('categorySentiment'), {
                type: 'bar',
                data: {
                    labels: catLabels,
                    datasets: [
                        { label: 'Positive', data: catLabels.map(c => categoryStats[c].positive), backgroundColor: '#10b981', borderRadius: 4 },
                        { label: 'Neutral',  data: catLabels.map(c => categoryStats[c].neutral),  backgroundColor: '#f59e0b', borderRadius: 4 },
                        { label: 'Negative', data: catLabels.map(c => categoryStats[c].negative), backgroundColor: '#ef4444', borderRadius: 4 },
                    ]
                },
                options: {
                    plugins: { legend: { labels: { color: '#9ca3af' } } },
                    scales: {
                        x: { stacked: true, ticks: { color: '#6b7280', maxRotation: 45 }, grid: { color: '#1f2937' } },
                        y: { stacked: true, ticks: { color: '#6b7280' }, grid: { color: '#1f2937' } }
                    }
                }
            });
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById('tab-' + tabName).classList.add('active');
        }

        // Load on page ready
        loadDashboard();
    </script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/data")
def api_data():
    """Serve aggregated analysis data as JSON for the frontend."""
    data = load_data()
    df = data["reviews"]

    if df.empty:
        return jsonify({"error": "No data. Run the pipeline first."}), 404

    # Basic stats
    total = len(df)
    pos_count = int((df.get("sentiment_label", pd.Series()) == "positive").sum())
    neg_count = int((df.get("sentiment_label", pd.Series()) == "negative").sum())
    neu_count = int((df.get("sentiment_label", pd.Series()) == "neutral").sum())

    topics_col = df.get("topic_id", pd.Series(dtype=int))
    valid_topics = topics_col[topics_col != -1] if not topics_col.empty else pd.Series(dtype=int)
    topics_count = valid_topics.nunique() if not valid_topics.empty else 0
    outlier_pct = round((topics_col == -1).mean() * 100, 1) if not topics_col.empty else 0

    avg_compound = round(df.get("sentiment_compound", pd.Series([0])).mean(), 4)

    stats = {
        "total_reviews": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "positive_pct": round(pos_count / total * 100, 1) if total else 0,
        "negative_pct": round(neg_count / total * 100, 1) if total else 0,
        "neutral_pct": round(neu_count / total * 100, 1) if total else 0,
        "topics_count": int(topics_count),
        "outlier_pct": outlier_pct,
        "avg_compound": avg_compound,
        "categories": int(df["category"].nunique()) if "category" in df.columns else 0,
    }

    # Histogram bins
    compound = df.get("sentiment_compound", pd.Series([0])).dropna()
    import numpy as np
    hist_counts, hist_edges = np.histogram(compound, bins=30)
    hist_bins = ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()

    # Star sentiment
    star_sentiment = {}
    if "star_rating" in df.columns and "sentiment_compound" in df.columns:
        for star in sorted(df["star_rating"].dropna().unique()):
            star_sentiment[str(int(star))] = round(
                df[df["star_rating"] == star]["sentiment_compound"].mean(), 4
            )

    # Category stats
    category_stats = {}
    if "category" in df.columns and "sentiment_label" in df.columns:
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            category_stats[cat] = {
                "positive": int((cat_df["sentiment_label"] == "positive").sum()),
                "negative": int((cat_df["sentiment_label"] == "negative").sum()),
                "neutral": int((cat_df["sentiment_label"] == "neutral").sum()),
            }

    return jsonify({
        "stats": stats,
        "histogram_bins": hist_bins,
        "histogram_counts": hist_counts.tolist(),
        "star_sentiment": star_sentiment,
        "category_stats": category_stats,
        "pain_points": data["pain_points"],
        "topics": data["topic_summary"],
    })


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    plots_dir = ROOT_DIR / config["dashboard"]["plots_dir"]
    return send_from_directory(str(plots_dir), filename)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    host = config["dashboard"]["host"]
    port = config["dashboard"]["port"]
    debug = config["dashboard"]["debug"]

    print(f"\n{'='*60}")
    print(f"  🔍 Customer Feedback Analyzer — Dashboard")
    print(f"  → http://{host}:{port}")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug)
