"""Interactive Web Dashboard & Live Testing Playground for IIT Jodhpur Academic Helpdesk.

Launch with:
    python dashboard.py

Open in browser:
    http://localhost:8080
"""
import csv
import json
import os
import sys
import time
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config as C
from helpdesk import Helpdesk

# Pre-initialize Helpdesk instances for fast interactive testing
_multi_hd = None
_single_hd = None

def get_helpdesk(mode="multi"):
    global _multi_hd, _single_hd
    if mode == "single":
        if _single_hd is None:
            _single_hd = Helpdesk(mode="single")
        return _single_hd
    else:
        if _multi_hd is None:
            _multi_hd = Helpdesk(mode="multi")
        return _multi_hd


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IIT Jodhpur Academic Helpdesk - Multi-Agent Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --primary-bg: #eff6ff;
            --success: #059669;
            --success-bg: #ecfdf5;
            --warning: #d97706;
            --warning-bg: #fffbeb;
            --danger: #dc2626;
            --danger-bg: #fef2f2;
            --surface: #ffffff;
            --bg: #f8fafc;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --radius: 12px;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: white;
            padding: 24px 32px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo-area { display: flex; align-items: center; gap: 16px; }
        .logo-icon {
            width: 44px;
            height: 44px;
            background: #3b82f6;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 700;
        }
        h1 { font-size: 22px; font-weight: 700; }
        .tagline { font-size: 13px; color: #94a3b8; font-weight: 400; margin-top: 2px; }
        .badge {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(147, 197, 253, 0.3);
            color: #93c5fd;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }

        .container {
            max-width: 1400px;
            margin: 24px auto;
            padding: 0 24px;
            flex: 1;
            width: 100%;
        }

        /* Tabs Navigation */
        .tabs {
            display: flex;
            gap: 8px;
            border-bottom: 2px solid var(--border);
            margin-bottom: 24px;
        }
        .tab-btn {
            background: none;
            border: none;
            padding: 12px 20px;
            font-size: 15px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .tab-btn:hover { color: var(--primary-light); }
        .tab-btn.active {
            color: var(--primary-light);
            border-bottom-color: var(--primary-light);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Stats Row */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: var(--surface);
            padding: 18px;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .metric-title { font-size: 13px; font-weight: 500; color: var(--text-muted); }
        .metric-value { font-size: 26px; font-weight: 700; margin-top: 6px; color: var(--primary); }
        .metric-sub { font-size: 12px; color: var(--success); margin-top: 4px; font-weight: 500; }

        /* Playground / Chat */
        .playground-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 900px) {
            .playground-grid { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--surface);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            padding: 24px;
        }
        .card-header {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .form-group { margin-bottom: 16px; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted); }
        select, textarea, input {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        select:focus, textarea:focus, input:focus { border-color: var(--primary-light); }
        textarea { resize: vertical; min-height: 90px; }

        .btn {
            background: var(--primary-light);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn:hover { background: #2563eb; }
        .btn:disabled { background: #94a3b8; cursor: not-allowed; }

        .presets {
            margin-top: 16px;
        }
        .presets-title { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 8px; }
        .presets-list { display: flex; flex-wrap: wrap; gap: 8px; }
        .preset-btn {
            background: var(--bg);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s;
            text-align: left;
        }
        .preset-btn:hover {
            border-color: var(--primary-light);
            background: var(--primary-bg);
            color: var(--primary-light);
        }

        /* Results Display */
        .response-box {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            min-height: 200px;
            white-space: pre-wrap;
            font-size: 14px;
            position: relative;
        }
        .agent-trace {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .agent-pill {
            background: #e0e7ff;
            color: #3730a3;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .trace-arrow { color: var(--text-muted); font-size: 12px; }

        .source-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 8px;
            margin-right: 6px;
        }

        .tools-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 12px;
        }
        .tools-table th, .tools-table td {
            border: 1px solid var(--border);
            padding: 6px 10px;
            text-align: left;
        }
        .tools-table th { background: #f8fafc; font-weight: 600; }

        /* Questions Table */
        .table-responsive {
            overflow-x: auto;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            background: white;
        }
        table.eval-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        table.eval-table th {
            background: #f8fafc;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
        }
        table.eval-table td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }
        table.eval-table tr:hover { background: #f8fafc; }

        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .status-pass { background: var(--success-bg); color: var(--success); }
        .status-fail { background: var(--danger-bg); color: var(--danger); }
        .status-refusal { background: #fef3c7; color: #b45309; }

        .filter-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            align-items: center;
        }
        .filter-btn {
            background: white;
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 500;
        }
        .filter-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .spinner {
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 3px solid white;
            width: 16px;
            height: 16px;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <header>
        <div class="header-content">
            <div class="logo-area">
                <div class="logo-icon" style="font-size: 14px; font-weight: 800; letter-spacing: 0.5px;">IITJ</div>
                <div>
                    <h1>IIT Jodhpur Academic Helpdesk</h1>
                    <div class="tagline">Autonomous Multi-Agent System (Google-ADK + ChromaDB + Ollama)</div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span class="badge">Architecture: Orchestrator + 3 Experts</span>
                <span class="badge" style="background: rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.3); color: #6ee7b7;">Status: Active</span>
            </div>
        </div>
    </header>

    <div class="container">
        <!-- Tabs Navigation -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('interactive')">Live Interactive Testing</button>
            <button class="tab-btn" onclick="switchTab('evaluation')">Evaluation Benchmark (28 Questions)</button>
            <button class="tab-btn" onclick="switchTab('escalation')">Human Review Queue (Logged Refusals)</button>
        </div>

        <!-- TAB 1: INTERACTIVE TESTING -->
        <div id="tab-interactive" class="tab-content active">
            <div class="playground-grid">
                <!-- Input Panel -->
                <div class="card">
                    <div class="card-header">
                        <span>Ask Academic Helpdesk</span>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <label style="margin: 0; font-size: 12px;">Agent Architecture:</label>
                            <select id="agentMode" style="width: auto; padding: 4px 8px; font-size: 12px;">
                                <option value="multi" selected>Multi-Agent (Orchestrator)</option>
                                <option value="single">Single-Agent Baseline</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="promptInput">Enter your query or request:</label>
                        <textarea id="promptInput" placeholder="e.g. How many Open Elective credits do I need to earn in order to get a B.Tech. Degree?"></textarea>
                    </div>

                    <button id="askBtn" class="btn" onclick="submitQuery()">
                        <span id="btnIcon"></span><span id="btnText">Ask Helpdesk</span>
                    </button>

                    <div class="presets">
                        <div class="presets-title">Quick Test Core Questions:</div>
                        <div class="presets-list">
                            <button class="preset-btn" onclick="setPreset('How many Open Elective credits do I need to earn in order to get a B.Tech. Degree?')"><b>[P03]</b> Open Elective Credits</button>
                            <button class="preset-btn" onclick="setPreset('What topics are covered in the Control Systems course of the Electrical Engineering curriculum?')"><b>[P09]</b> EE Control Systems Topics</button>
                            <button class="preset-btn" onclick="setPreset('What is the attendance requirement to sit in the Major exam?')"><b>[P01]</b> Major Exam Attendance</button>
                            <button class="preset-btn" onclick="setPreset('What is the slot for the course Autonomous Systems?')"><b>[S01]</b> Autonomous Systems Slot</button>
                            <button class="preset-btn" onclick="setPreset('How many 400 level electives have been floated by the Humanities department in this semester?')"><b>[S05]</b> 400-level Humanities Electives</button>
                            <button class="preset-btn" onclick="setPreset('Write an email to the course instructor of CSL7570 that I need leave for 2 days')"><b>[E01]</b> Sick Leave Email (CSL7570)</button>
                            <button class="preset-btn" onclick="setPreset('What are the available dining options on the campus?')"><b>[O01]</b> Campus Dining (Out of Scope)</button>
                            <button class="preset-btn" onclick="setPreset('What is the best way to reach the campus from Jodhpur city center?')"><b>[O02]</b> Travel Route (Out of Scope)</button>
                        </div>
                    </div>
                </div>

                <!-- Response Panel -->
                <div class="card">
                    <div class="card-header">
                        <span>Agent Response & Execution Trace</span>
                        <span id="execTime" style="font-size: 12px; color: var(--text-muted);">0.0s</span>
                    </div>

                    <div class="agent-trace" id="agentTrace">
                        <span style="font-size: 12px; color: var(--text-muted);">Awaiting query...</span>
                    </div>

                    <div class="response-box" id="responseBox">
Ask any question on the left to see live multi-agent reasoning, RAG chunk retrieval, tool calls, and citations.
                    </div>

                    <div id="sourcesArea" style="margin-top: 12px;"></div>
                    <div id="toolsArea" style="margin-top: 12px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 2: EVALUATION BENCHMARK -->
        <div id="tab-evaluation" class="tab-content">
            <div class="card" style="margin-bottom: 20px;">
                <div class="card-header">
                    <span>Benchmark Overview: 28 Questions Across 4 Categories</span>
                    <div style="display: flex; gap: 8px;">
                        <a href="/download/excel" target="_blank" class="btn" style="padding: 6px 14px; font-size: 13px; text-decoration: none; background: #059669;">Download Excel Workbook (.xlsx)</a>
                        <a href="/download/csv" target="_blank" class="btn" style="padding: 6px 14px; font-size: 13px; text-decoration: none; background: #3b82f6;">Download Graded CSV</a>
                    </div>
                </div>
                <div class="filter-bar">
                    <button class="filter-btn active" onclick="filterCategory('all', this)">All (28)</button>
                    <button class="filter-btn" onclick="filterCategory('policy', this)">Policy & Curriculum (9)</button>
                    <button class="filter-btn" onclick="filterCategory('schedule', this)">Schedule & Timetable (8)</button>
                    <button class="filter-btn" onclick="filterCategory('email', this)">Email Drafting (5)</button>
                    <button class="filter-btn" onclick="filterCategory('out_of_scope', this)">Out of Scope Refusals (6)</button>
                </div>

                <div class="table-responsive">
                    <table class="eval-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Category</th>
                                <th>Question Prompt</th>
                                <th>Expected Fact / Criteria</th>
                                <th>Generated Answer</th>
                                <th>Cited Source</th>
                                <th>Accuracy</th>
                                <th>Grounded</th>
                            </tr>
                        </thead>
                        <tbody id="evalTableBody">
                            <tr><td colspan="8" style="text-align: center; padding: 20px;">Loading evaluation records...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 3: HUMAN REVIEW ESCALATION QUEUE -->
        <div id="tab-escalation" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span>Unanswered Questions Logged for Human Review</span>
                    <span class="badge" style="background: rgba(220, 38, 38, 0.1); color: var(--danger); border-color: rgba(220, 38, 38, 0.2);">
                        Logged to: unanswered_questions.jsonl
                    </span>
                </div>
                <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
                    When out-of-scope questions arrive (dining, travel, city information, etc.), the Helpdesk politely refuses with "I do not know" and logs the query here so administrators can provide human assistance.
                </p>
                <div class="table-responsive">
                    <table class="eval-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Question</th>
                                <th>Escalation Reason</th>
                                <th>Originating Agent</th>
                            </tr>
                        </thead>
                        <tbody id="unansweredTableBody">
                            <tr><td colspan="4" style="text-align: center; padding: 20px;">Loading escalation queue...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(name) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + name).classList.add('active');
            if (name === 'evaluation') loadEvaluation();
            if (name === 'escalation') loadUnanswered();
        }

        function setPreset(text) {
            document.getElementById('promptInput').value = text;
        }

        async function submitQuery() {
            const prompt = document.getElementById('promptInput').value.trim();
            if (!prompt) return;

            const mode = document.getElementById('agentMode').value;
            const btn = document.getElementById('askBtn');
            const btnIcon = document.getElementById('btnIcon');
            const btnText = document.getElementById('btnText');
            const resBox = document.getElementById('responseBox');
            const traceBox = document.getElementById('agentTrace');
            const execTime = document.getElementById('execTime');
            const sourcesArea = document.getElementById('sourcesArea');
            const toolsArea = document.getElementById('toolsArea');

            btn.disabled = true;
            btnIcon.innerHTML = '<span class="spinner"></span>';
            btnText.innerText = 'Consulting Agents...';
            resBox.innerText = 'Invoking Helpdesk orchestrator and specialized agent tools...';
            sourcesArea.innerHTML = '';
            toolsArea.innerHTML = '';

            const t0 = performance.now();
            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: prompt, mode: mode })
                });
                const data = await res.json();
                const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
                execTime.innerText = elapsed + 's';

                // Display Answer
                resBox.innerText = data.answer || '(No response text)';

                // Display Trace
                if (data.agents && data.agents.length > 0) {
                    let traceHtml = '<span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">Agent Route:</span> ';
                    data.agents.forEach((ag, idx) => {
                        traceHtml += `<span class="agent-pill">${ag}</span>`;
                        if (idx < data.agents.length - 1) traceHtml += '<span class="trace-arrow">&rarr;</span>';
                    });
                    if (data.refused) {
                        traceHtml += ' <span class="status-badge status-refusal">Politely Refused & Logged</span>';
                    }
                    traceBox.innerHTML = traceHtml;
                } else {
                    traceBox.innerHTML = '<span class="agent-pill">Single Agent</span>';
                }

                // Display Sources Cited
                if (data.cited_sources && data.cited_sources.length > 0) {
                    let srcHtml = '<div style="font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 4px;">Verified Document Citations:</div>';
                    data.cited_sources.forEach(src => {
                        srcHtml += `<span class="source-pill">${src}</span>`;
                    });
                    sourcesArea.innerHTML = srcHtml;
                }

                // Display Tools
                if (data.tool_calls && data.tool_calls.length > 0) {
                    let tbl = '<div style="font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 4px;">Tools Executed:</div>';
                    tbl += '<table class="tools-table"><tr><th>Tool Name</th><th>Arguments</th></tr>';
                    data.tool_calls.forEach(tc => {
                        tbl += `<tr><td><b>${tc.name}</b></td><td><code>${JSON.stringify(tc.args)}</code></td></tr>`;
                    });
                    tbl += '</table>';
                    toolsArea.innerHTML = tbl;
                }

            } catch (err) {
                resBox.innerText = 'Error calling Helpdesk API: ' + err.message;
            } finally {
                btn.disabled = false;
                btnIcon.innerHTML = '';
                btnText.innerText = 'Ask Helpdesk';
            }
        }

        let allRecords = [];
        async function loadEvaluation() {
            try {
                const res = await fetch('/api/results');
                const data = await res.json();
                allRecords = data.records || [];
                renderEvalTable(allRecords);
            } catch (err) {
                document.getElementById('evalTableBody').innerHTML = `<tr><td colspan="8" style="color:red;">Failed to load evaluation data: ${err.message}</td></tr>`;
            }
        }

        function filterCategory(cat, btn) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (cat === 'all') {
                renderEvalTable(allRecords);
            } else {
                renderEvalTable(allRecords.filter(r => r.category === cat));
            }
        }

        function renderEvalTable(records) {
            const tbody = document.getElementById('evalTableBody');
            if (!records || records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px;">No records found. Run run_eval.py first.</td></tr>';
                return;
            }
            tbody.innerHTML = records.map(r => {
                const accBadge = r.accuracy === true 
                    ? '<span class="status-badge status-pass">PASS (100%)</span>' 
                    : (r.refused ? '<span class="status-badge status-refusal">REFUSED (100%)</span>' : '<span class="status-badge status-fail">FAIL</span>');
                const gndBadge = r.grounded 
                    ? '<span class="status-badge status-pass">VERIFIED</span>' 
                    : (r.category === 'out_of_scope' ? '<span class="status-badge status-refusal">N/A</span>' : '<span class="status-badge status-fail">UNVERIFIED</span>');

                return `<tr>
                    <td><b>${r.id}</b></td>
                    <td><span class="badge" style="background:#e2e8f0; color:#334155; border:none;">${r.category}</span></td>
                    <td style="max-width: 280px;">${r.question}</td>
                    <td style="max-width: 200px; font-size: 12px; color: #475569;">${r.expected || r.gold || 'Criteria verified'}</td>
                    <td style="max-width: 320px; font-size: 12px;">${r.answer || ''}</td>
                    <td style="font-size: 11px; font-family: monospace;">${r.source || 'None'}</td>
                    <td>${accBadge}</td>
                    <td>${gndBadge}</td>
                </tr>`;
            }).join('');
        }

        async function loadUnanswered() {
            try {
                const res = await fetch('/api/unanswered');
                const list = await res.json();
                const tbody = document.getElementById('unansweredTableBody');
                if (!list || list.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 20px;">No unanswered questions logged yet.</td></tr>';
                    return;
                }
                tbody.innerHTML = list.map(item => `<tr>
                    <td style="font-size: 12px; color: var(--text-muted);">${item.timestamp || 'Recent'}</td>
                    <td><b>${item.question || ''}</b></td>
                    <td><span class="status-badge status-refusal">${item.reason || 'out of scope'}</span></td>
                    <td><span class="agent-pill">${item.agent || 'helpdesk_orchestrator'}</span></td>
                </tr>`).join('');
            } catch (err) {
                document.getElementById('unansweredTableBody').innerHTML = `<tr><td colspan="4" style="color:red;">Error: ${err.message}</td></tr>`;
            }
        }
    </script>
</body>
</html>
"""


class HelpdeskServer(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/questions":
            q_file = C.BASE_DIR / "eval" / "questions.json"
            if q_file.exists():
                data = json.load(open(q_file, encoding="utf-8"))
            else:
                data = []
            self._send_json(data)
            return

        if path == "/download/excel":
            excel_path = C.BASE_DIR / "results" / "Academic_Helpdesk_Evaluation_Metrics.xlsx"
            if not excel_path.exists():
                self.send_error(404, "Excel report not found. Run summarize.py first.")
                return
            content = excel_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", 'attachment; filename="Academic_Helpdesk_Evaluation_Metrics.xlsx"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/download/csv":
            csv_path = C.BASE_DIR / "results" / "graded.csv"
            if not csv_path.exists():
                csv_path = C.BASE_DIR / "results" / "graded_latest.csv"
            if not csv_path.exists():
                self.send_error(404, "CSV report not found. Run summarize.py first.")
                return
            content = csv_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="graded.csv"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/results":
            res_dir = C.BASE_DIR / "results"
            csv_path = res_dir / "graded.csv" if (res_dir / "graded.csv").exists() else res_dir / "graded_latest.csv"
            records = []
            if csv_path.exists():
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # concise view: multi-agent run 1
                        if str(row.get("run")) == "1" and "multi" in str(row.get("mode")):
                            records.append({
                                "id": row.get("question_id"),
                                "category": row.get("category"),
                                "question": row.get("question"),
                                "expected": row.get("expected_answer_criteria"),
                                "answer": row.get("generated_answer"),
                                "source": row.get("sources_cited"),
                                "accuracy": row.get("accuracy") == "PASS",
                                "grounded": row.get("grounded") == "PASS",
                                "refused": row.get("refused") == "YES"
                            })
            self._send_json({"records": records})
            return

        if path == "/api/unanswered":
            log_file = C.BASE_DIR / "unanswered_questions.jsonl"
            items = []
            if log_file.exists():
                lines = log_file.read_text(encoding="utf-8").splitlines()
                for line in lines[-20:]:  # last 20
                    if line.strip():
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            pass
            self._send_json(list(reversed(items)))
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/ask":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            question = data.get("question", "")
            mode = data.get("mode", "multi")

            hd = get_helpdesk(mode)
            result = hd.ask(question)
            
            # Extract cited sources
            ans = result.get("answer", "")
            known_files = sorted({p.name for p in C.PDF_DIR.rglob("*.pdf")} | {C.TIMETABLE_CSV.name, C.FACULTY_CSV.name})
            result["cited_sources"] = [f for f in known_files if f.lower() in ans.lower()]

            self._send_json(result)
            return

        self.send_error(404, "Not Found")

    def _send_json(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # quiet stdout logging


def main():
    port = 8501
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    server = ThreadingHTTPServer(("127.0.0.1", port), HelpdeskServer)
    print(f"=================================================================")
    print(f"  IIT Jodhpur Academic Helpdesk Interactive Dashboard Active!    ")
    print(f"  Open in your web browser: http://localhost:{port}             ")
    print(f"=================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
        server.server_close()


if __name__ == "__main__":
    main()
