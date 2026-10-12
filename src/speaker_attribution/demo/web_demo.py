"""
EchoLocate — Real Human Speaker Attribution Web Demonstration
=============================================================
Self-contained, lightweight FastAPI web server providing:
  - In-browser microphone recording (laptop & smartphone support via Web Audio API)
  - Multi-sample enrollment with audio quality gates (speech activity, RMS, SNR, duration, clipping)
  - Local WAV persistence under demo_data/speakers/<speaker_id>/recordings/*.wav
  - Persistent speaker profile management (List, Add Sample, View Recordings, Delete)
  - Instant live speaker identification card with explicit UNKNOWN preservation
  - Streaming live identification mode with temporal stability tracking
  - Interactive Blind Test Mode with ground-truth blinding and scorecard summary
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from ..adapters.audio_source_adapter import AudioSourceAdapter
from ..encoders.factory import create_encoder
from ..preprocessing.audio_processor import AudioProcessor
from .session import DemonstrationSession

logger = logging.getLogger("EchoLocate.SpeakerDemo")


def create_demo_app(
    db_dir: str = "demo_data/speakers",
    model_name: str = "campp",
    default_threshold: float = 0.42,
) -> FastAPI:
    """Instantiate FastAPI demo application."""
    app = FastAPI(title="EchoLocate Speaker Attribution Demo")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session = DemonstrationSession(
        db_dir=db_dir,
        model_name=model_name,
        default_threshold=default_threshold,
    )
    processor = AudioProcessor(target_sample_rate=16000)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return get_demo_html()

    # -------------------------------------------------------------
    # Speaker Management & Queries
    # -------------------------------------------------------------
    @app.get("/api/speakers")
    @app.get("/speaker/list")
    def get_speakers():
        speakers = []
        for s_id in session.db.get_enrolled_speakers():
            prof = session.db.get_profile(s_id)
            d_name = prof.metadata.get("display_name", s_id) if (prof and prof.metadata) else s_id
            recs = prof.metadata.get("recordings", []) if (prof and prof.metadata) else []
            speakers.append({
                "speaker_id": s_id,
                "display_name": d_name,
                "sample_count": prof.sample_count if prof else 0,
                "model_name": prof.model_name if prof else "",
                "created_at": prof.created_at if prof else 0,
                "last_updated": prof.metadata.get("last_updated", prof.created_at) if (prof and prof.metadata) else 0,
                "recordings_count": len(recs),
            })
        return {
            "speakers": speakers,
            "active_model": session.model_name,
            "threshold": session.matcher.default_threshold,
            "storage_path": str(Path(db_dir).resolve()),
        }

    @app.get("/api/speaker/{speaker_id}")
    @app.get("/speaker/{speaker_id}")
    def get_speaker_profile(speaker_id: str):
        details = session.get_speaker_details(speaker_id)
        if not details:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
        return details

    @app.delete("/api/speaker/{speaker_id}")
    @app.delete("/speaker/{speaker_id}")
    def delete_speaker_endpoint(speaker_id: str):
        deleted = session.delete_speaker(speaker_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
        return {"status": "deleted", "speaker_id": speaker_id}

    # -------------------------------------------------------------
    # Enrollment
    # -------------------------------------------------------------
    @app.post("/api/enroll")
    @app.post("/speaker/enroll/sample")
    async def enroll_sample_endpoint(
        speaker_id: str = Form(...),
        display_name: str = Form(""),
        audio_file: UploadFile = File(...),
        override_quality: bool = Form(False),
    ):
        clean_id = speaker_id.strip()
        if not clean_id:
            raise HTTPException(status_code=400, detail="Speaker ID required")

        d_name = display_name.strip() or clean_id
        content = await audio_file.read()
        try:
            raw_data, sr = AudioSourceAdapter.from_bytes(content)
            orig_samples = len(raw_data)
            orig_channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]
            seg = processor.process(raw_data, sample_rate=sr)
            audio = seg.samples
            logger.info(
                "[Phase 4 Audio Audit - Enroll] Browser SR: %d Hz | Channels: %d | Captured Samples: %d | "
                "Resample Ratio: %.3f | Final SR: %d Hz | Final Duration: %.2fs",
                sr,
                orig_channels,
                orig_samples,
                16000.0 / sr,
                seg.sample_rate,
                seg.duration_sec,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")

        ok, quality, prof = session.enroll_sample(
            speaker_id=clean_id,
            display_name=d_name,
            audio=audio,
            allow_override_quality=override_quality,
        )

        return {
            "success": ok,
            "quality": {
                "passed": quality.passed,
                "reason": quality.status_reason,
                "duration_sec": round(quality.total_duration_sec, 2),
                "speech_duration_sec": round(quality.speech_duration_sec, 2),
                "speech_ratio": round(quality.speech_ratio, 2),
                "rms": round(quality.rms, 4),
                "snr_db": round(quality.snr_db, 1),
                "is_clipped": quality.is_clipped,
            },
            "profile": {
                "speaker_id": prof.speaker_id if prof else clean_id,
                "display_name": d_name,
                "sample_count": prof.sample_count if prof else 0,
                "last_sample_file": prof.metadata.get("last_sample_file") if (prof and prof.metadata) else None,
            } if prof else None,
        }

    # -------------------------------------------------------------
    # Identification
    # -------------------------------------------------------------
    @app.post("/api/identify")
    @app.post("/speaker/identify")
    async def identify_endpoint(
        audio_file: UploadFile = File(...),
        threshold: Optional[float] = Form(None),
    ):
        content = await audio_file.read()
        try:
            raw_data, sr = AudioSourceAdapter.from_bytes(content)
            orig_samples = len(raw_data)
            orig_channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]
            seg = processor.process(raw_data, sample_rate=sr)
            audio = seg.samples
            logger.info(
                "[Phase 4 Audio Audit - Identify] Browser SR: %d Hz | Channels: %d | Captured Samples: %d | "
                "Resample Ratio: %.3f | Final SR: %d Hz | Final Duration: %.2fs",
                sr,
                orig_channels,
                orig_samples,
                16000.0 / sr,
                seg.sample_rate,
                seg.duration_sec,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")

        if threshold is not None:
            session.matcher.default_threshold = threshold

        res = session.identify_sample(audio, use_temporal_tracking=False)
        return res

    @app.post("/api/speaker/live/identify")
    async def live_identify_endpoint(
        audio_file: UploadFile = File(...),
        threshold: Optional[float] = Form(None),
    ):
        """Streaming chunk identification with temporal tracker smoothing."""
        content = await audio_file.read()
        try:
            raw_data, sr = AudioSourceAdapter.from_bytes(content)
            orig_samples = len(raw_data)
            orig_channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]
            seg = processor.process(raw_data, sample_rate=sr)
            audio = seg.samples
            logger.info(
                "[Phase 4 Audio Audit - Live Stream] Browser SR: %d Hz | Channels: %d | Captured Samples: %d | "
                "Resample Ratio: %.3f | Final SR: %d Hz | Final Duration: %.2fs",
                sr,
                orig_channels,
                orig_samples,
                16000.0 / sr,
                seg.sample_rate,
                seg.duration_sec,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")

        if threshold is not None:
            session.matcher.default_threshold = threshold

        res = session.identify_sample(audio, use_temporal_tracking=True)
        return res

    # -------------------------------------------------------------
    # Blind Testing
    # -------------------------------------------------------------
    @app.post("/api/blind-test/record")
    @app.post("/speaker/blind-test/record")
    async def record_blind_test_endpoint(
        true_speaker: str = Form(...),
        condition: str = Form("normal"),
        audio_file: UploadFile = File(...),
    ):
        content = await audio_file.read()
        try:
            raw_data, sr = AudioSourceAdapter.from_bytes(content)
            seg = processor.process(raw_data, sample_rate=sr)
            audio = seg.samples
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")

        test_case = session.record_blind_test(
            true_speaker=true_speaker.strip(),
            audio=audio,
            condition=condition.strip(),
        )
        return {
            "test_id": test_case.test_id,
            "status": "recorded",
            "condition": test_case.condition,
        }

    @app.get("/api/blind-test/summary")
    @app.get("/speaker/blind-test/summary")
    def get_blind_test_summary_endpoint():
        summary = session.compute_summary()
        return summary

    @app.post("/api/blind-test/reset")
    @app.post("/speaker/blind-test/reset")
    def reset_blind_test_endpoint():
        session.blind_tests.clear()
        return {"status": "cleared"}

    # -------------------------------------------------------------
    # Model Selection
    # -------------------------------------------------------------
    @app.post("/api/model/switch")
    @app.post("/speaker/model/switch")
    def switch_model_endpoint(model_name: str = Form(...)):
        if model_name not in ["campp", "eres2netv2", "ecapa"]:
            raise HTTPException(status_code=400, detail="Invalid model")
        session.switch_model(model_name)
        return {"active_model": session.model_name}

    return app


def get_demo_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EchoLocate — Persistent Speaker Attribution System</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #121826;
      --card-alt: #0c101b;
      --border: #1f293d;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text: #f1f5f9;
      --text-dim: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    .container { max-width: 1120px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 18px; }
    h1 { font-size: 24px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
    .badge { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
    .badge-cyan { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid var(--accent); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid var(--warning); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid var(--danger); }
    
    .nav-tabs { display: flex; gap: 8px; margin-bottom: 22px; flex-wrap: wrap; }
    .tab-btn { background: var(--card); border: 1px solid var(--border); color: var(--text-dim); padding: 11px 22px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.2s; }
    .tab-btn:hover { color: #fff; border-color: var(--accent); }
    .tab-btn.active { background: var(--accent); color: #000; border-color: var(--accent); }
    
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
    .card h2 { font-size: 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    
    label { display: block; font-size: 13px; font-weight: 500; color: var(--text-dim); margin-bottom: 6px; }
    input, select { width: 100%; padding: 10px 12px; background: var(--card-alt); border: 1px solid var(--border); border-radius: 6px; color: #fff; margin-bottom: 14px; font-size: 14px; }
    input:focus, select:focus { outline: none; border-color: var(--accent); }
    
    .btn { background: var(--accent); color: #000; border: none; padding: 12px 20px; border-radius: 8px; font-weight: 700; cursor: pointer; width: 100%; display: flex; justify-content: center; align-items: center; gap: 8px; transition: all 0.2s; font-size: 14px; }
    .btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
    .btn-secondary { background: var(--card-alt); color: var(--text); border: 1px solid var(--border); }
    .btn-danger { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); width: auto; padding: 6px 12px; font-size: 12px; }
    .btn-danger:hover { background: var(--danger); color: #fff; }
    
    .btn-record { background: #ef4444; color: #fff; }
    .btn-record.recording { background: #dc2626; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); } 70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
    
    canvas.waveform { width: 100%; height: 50px; background: var(--card-alt); border-radius: 6px; border: 1px solid var(--border); margin-bottom: 12px; }
    .meter-container { height: 8px; background: var(--card-alt); border-radius: 4px; overflow: hidden; margin-bottom: 16px; border: 1px solid var(--border); }
    .meter-bar { height: 100%; width: 0%; background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); transition: width 0.05s linear; }
    
    .result-box { border-radius: 10px; padding: 22px; margin-top: 16px; }
    .result-known { background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.4); }
    .result-unknown { background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.4); }
    
    .result-title { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
    .metric-row { display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px; }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #fff; }
    
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }
    th { color: var(--text-dim); font-weight: 600; }
    
    .prompt-box { background: rgba(56, 189, 248, 0.08); border-left: 3px solid var(--accent); padding: 12px 14px; border-radius: 4px; font-size: 13px; margin-bottom: 16px; line-height: 1.5; }
    .quality-chip { display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; margin-right: 6px; margin-top: 4px; }
    .chip-ok { background: rgba(16, 185, 129, 0.15); color: var(--success); }
    .chip-warn { background: rgba(245, 158, 11, 0.15); color: var(--warning); }
    .chip-fail { background: rgba(239, 68, 68, 0.15); color: var(--danger); }
    
    .privacy-notice { margin-top: 24px; padding: 12px 16px; background: rgba(255, 255, 255, 0.02); border: 1px solid var(--border); border-radius: 8px; font-size: 12px; color: var(--text-dim); display: flex; align-items: center; justify-content: space-between; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎙️ EchoLocate <span style="font-weight:400; color:var(--text-dim); font-size:18px;">Speaker Attribution</span></h1>
        <p style="font-size:13px; color:var(--text-dim); margin-top:4px;">Persistent Real-Speaker Enrollment, Identification & Strict UNKNOWN Gating</p>
      </div>
      <div style="display:flex; align-items:center; gap:12px;">
        <select id="model-select" onchange="switchModel()" style="margin-bottom:0; width:220px;">
          <option value="campp" selected>Model: CAM++</option>
          <option value="eres2netv2">Model: ERes2NetV2</option>
          <option value="ecapa">Model: ECAPA-TDNN (VoxCeleb)</option>
        </select>
        <span class="badge badge-cyan" id="db-status">Active Profiles: 0</span>
      </div>
    </header>

    <div class="nav-tabs">
      <button class="tab-btn active" onclick="showTab('tab-identify')">1. Identify Speaker</button>
      <button class="tab-btn" onclick="showTab('tab-enroll')">2. Add New Speaker</button>
      <button class="tab-btn" onclick="showTab('tab-management')">3. Speaker Management (<span id="tab-count">0</span>)</button>
      <button class="tab-btn" onclick="showTab('tab-blind')">4. Blind Evaluation Mode</button>
    </div>

    <!-- TAB 1: IDENTIFICATION -->
    <div id="tab-identify" class="tab-content active">
      <div class="grid-2">
        <div class="card">
          <h2>
            <span>Microphone Capture</span>
            <span id="ident-status-badge" class="badge badge-cyan">Ready</span>
          </h2>
          <p style="font-size:13px; color:var(--text-dim); margin-bottom:14px;">
            Speak a sentence into your laptop or phone microphone. The system extracts your neural embedding and compares it against enrolled centroids.
          </p>

          <canvas id="canvas-ident" class="waveform"></canvas>
          <div class="meter-container"><div id="meter-ident" class="meter-bar"></div></div>

          <div style="display:flex; gap:10px; margin-bottom:14px;">
            <button id="btn-record-ident" class="btn btn-record" onclick="recordAndIdentify()">
              <span>🔴 Record & Identify (3.0s)</span>
            </button>
            <button id="btn-live-stream" class="btn btn-secondary" onclick="toggleLiveStream()" style="width:auto; min-width:140px;">
              <span>⚡ Live Stream</span>
            </button>
          </div>

          <div style="margin-top:10px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; color:var(--text-dim); margin-bottom:4px;">
              <span>UNKNOWN Threshold Boundary</span>
              <span id="thresh-val" style="font-family:'JetBrains Mono'; font-weight:600; color:#fff;">0.42</span>
            </div>
            <input type="range" id="thresh-slider" min="0.20" max="0.85" step="0.01" value="0.42" oninput="updateThresh(this.value)" style="margin-bottom:0;">
          </div>
        </div>

        <div class="card">
          <h2>Attribution Result</h2>
          <div id="ident-empty" style="padding:46px 20px; text-align:center; color:var(--text-dim); font-size:14px;">
            No acoustic event processed.<br>Click <strong>Record & Identify</strong> or <strong>Live Stream</strong> to test.
          </div>

          <div id="ident-card" class="result-box" style="display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span id="badge-decision" class="badge"></span>
              <span id="res-latency" style="font-family:'JetBrains Mono'; font-size:12px; color:var(--text-dim);"></span>
            </div>
            <div id="res-speaker" class="result-title" style="margin-top:8px;"></div>
            
            <div class="metric-row">
              <span>Similarity Score:</span>
              <span id="res-sim" class="metric-val"></span>
            </div>
            <div class="metric-row">
              <span>Calibrated Confidence:</span>
              <span id="res-conf" class="metric-val"></span>
            </div>
            <div class="metric-row">
              <span>Margin over Runner-up:</span>
              <span id="res-margin" class="metric-val"></span>
            </div>
            <div class="metric-row" id="row-temporal" style="display:none;">
              <span>Temporal Stability:</span>
              <span id="res-stability" class="metric-val"></span>
            </div>

            <h4 style="font-size:12px; color:var(--text-dim); margin-top:18px; text-transform:uppercase;">Candidate Rankings</h4>
            <table id="table-alts">
              <thead><tr><th>Rank</th><th>Candidate</th><th>Similarity</th></tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: ENROLLMENT -->
    <div id="tab-enroll" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <h2>Add New Speaker</h2>
          
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
            <div>
              <label>Person's Name (e.g. Rahul, Priya)</label>
              <input type="text" id="enroll-name" placeholder="Rahul" oninput="syncId()">
            </div>
            <div>
              <label>Speaker ID</label>
              <input type="text" id="enroll-id" placeholder="rahul">
            </div>
          </div>

          <div class="prompt-box">
            <strong>Sample <span id="enroll-step">1</span> of <span id="enroll-total">5</span>:</strong><br>
            <span id="enroll-prompt" style="font-size:14px; font-weight:600; color:#fff;">"Say your name and where you are from."</span>
            <div style="font-size:11px; color:var(--text-dim); margin-top:4px;">(Varied sentences ensure the system learns voice traits, not specific phrases)</div>
          </div>

          <canvas id="canvas-enroll" class="waveform"></canvas>
          <div class="meter-container"><div id="meter-enroll" class="meter-bar"></div></div>

          <div style="display:flex; gap:10px;">
            <button id="btn-record-enroll" class="btn btn-record" onclick="recordEnrollmentSample()">
              <span>🔴 Record Sample (<span id="enroll-step-btn">1</span>/5)</span>
            </button>
          </div>

          <div id="enroll-quality-card" style="margin-top:16px; display:none; background:var(--card-alt); border:1px solid var(--border); border-radius:8px; padding:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
              <span style="font-size:13px; font-weight:600;">Quality Diagnostics</span>
              <span id="quality-verdict-badge" class="badge"></span>
            </div>
            <div id="quality-chips"></div>
          </div>
        </div>

        <div class="card">
          <h2>Persistent Enrollment Protocol</h2>
          <ul style="font-size:13px; line-height:2.2; color:var(--text-dim); padding-left:18px;">
            <li>✓ <strong>Multi-Sample Centroid:</strong> Collects 5 independent utterances per speaker.</li>
            <li>✓ <strong>Strict Quality Verification:</strong> Rejects silence, short speech (< 0.8s), low SNR (< 6dB), and audio clipping.</li>
            <li>✓ <strong>Local Audio Storage:</strong> Accepted recordings are permanently saved to <code style="color:var(--accent);">demo_data/speakers/&lt;id&gt;/recordings/*.wav</code>.</li>
            <li>✓ <strong>Centroid Representation:</strong> Accumulates embeddings into an updated unit-normalized centroid vector.</li>
            <li>✓ <strong>Survives Restarts:</strong> Profiles & recordings persist across application restarts.</li>
          </ul>

          <div style="margin-top:20px; padding:12px; background:var(--card-alt); border-radius:6px; border:1px solid var(--border);">
            <div style="font-size:12px; color:var(--text-dim);">ACTIVE STORAGE DIRECTORY</div>
            <div id="storage-dir-display" style="font-family:'JetBrains Mono'; font-size:11px; color:var(--accent); word-break:break-all; margin-top:4px;">...</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: SPEAKER MANAGEMENT -->
    <div id="tab-management" class="tab-content">
      <div class="card">
        <h2>
          <span>Enrolled Speaker Profiles</span>
          <button class="btn btn-secondary" onclick="loadSpeakers()" style="width:auto; padding:6px 14px; font-size:13px;">🔄 Refresh</button>
        </h2>
        
        <table id="table-speakers-mgmt">
          <thead>
            <tr>
              <th>Name</th>
              <th>ID</th>
              <th>Accepted Samples</th>
              <th>Model</th>
              <th>Recordings Stored</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
        
        <div id="mgmt-empty" style="padding:40px; text-align:center; color:var(--text-dim); display:none;">
          No speakers enrolled yet. Click <strong>Add New Speaker</strong> to enroll real people.
        </div>
      </div>
    </div>

    <!-- TAB 4: BLIND TEST -->
    <div id="tab-blind" class="tab-content">
      <div class="card">
        <h2>
          <span>Blind Evaluation Testing</span>
          <button class="btn btn-secondary" onclick="resetBlindTest()" style="width:auto; padding:6px 14px; font-size:12px;">Reset Trials</button>
        </h2>
        <p style="font-size:13px; color:var(--text-dim); margin-bottom:16px;">
          The evaluator enters the ground-truth name (or 'unknown' for an unregistered person). The prediction remains hidden until scorecard review.
        </p>

        <div style="display:grid; grid-template-columns: 2fr 1fr 2fr; gap:16px; margin-bottom:16px;">
          <div>
            <label>Ground Truth Speaker</label>
            <input type="text" id="blind-true" placeholder="e.g. rahul or unknown">
          </div>
          <div>
            <label>Condition</label>
            <select id="blind-cond">
              <option value="normal">Normal (0.5m)</option>
              <option value="close">Close Mic (0.1m)</option>
              <option value="1m">Far (1.0m)</option>
              <option value="2m">Far (2.0m)</option>
              <option value="diff_sentence">Different Sentence</option>
              <option value="noise">Ambient Noise</option>
            </select>
          </div>
          <div>
            <label>&nbsp;</label>
            <button class="btn btn-record" onclick="recordBlindTestTrial()"><span>🔴 Record Blind Trial (3.0s)</span></button>
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px;">
          <h3 style="font-size:16px;">Session Evaluation Scorecard</h3>
          <button class="btn" style="width:auto; padding:8px 18px;" onclick="loadBlindSummary()">Reveal Scorecard</button>
        </div>

        <div id="blind-scorecard" style="margin-top:16px; display:none;">
          <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-bottom:16px;">
            <div style="background:var(--card-alt); padding:16px; border-radius:8px; border:1px solid var(--border);">
              <div style="color:var(--text-dim); font-size:12px;">KNOWN ACCURACY</div>
              <div id="sum-known-acc" style="font-size:24px; font-weight:700; color:var(--success);">0%</div>
            </div>
            <div style="background:var(--card-alt); padding:16px; border-radius:8px; border:1px solid var(--border);">
              <div style="color:var(--text-dim); font-size:12px;">UNKNOWN REJECTION</div>
              <div id="sum-unk-rej" style="font-size:24px; font-weight:700; color:var(--warning);">0%</div>
            </div>
            <div style="background:var(--card-alt); padding:16px; border-radius:8px; border:1px solid var(--border);">
              <div style="color:var(--text-dim); font-size:12px;">OVERALL ACCURACY</div>
              <div id="sum-all-acc" style="font-size:24px; font-weight:700; color:var(--accent);">0%</div>
            </div>
            <div style="background:var(--card-alt); padding:16px; border-radius:8px; border:1px solid var(--border);">
              <div style="color:var(--text-dim); font-size:12px;">MEAN LATENCY</div>
              <div id="sum-lat" style="font-size:24px; font-weight:700; color:#fff;">0 ms</div>
            </div>
          </div>

          <table id="table-blind-trials">
            <thead>
              <tr><th>#</th><th>True Label</th><th>Predicted</th><th>Status</th><th>Sim</th><th>Conf</th><th>Condition</th><th>Verdict</th></tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="privacy-notice">
      <span>🔒 <strong>Local Privacy Guaranteed:</strong> Voice recordings (.wav) and biometric centroids are stored locally under <code style="color:var(--accent);">demo_data/speakers/</code>. Zero cloud API calls.</span>
      <span id="live-indicator" style="display:none;" class="badge badge-red">● LIVE STREAM ACTIVE</span>
    </div>
  </div>

  <script>
    let currentStep = 1;
    const totalSteps = 5;
    let isLiveStreaming = false;
    let liveInterval = null;

    const samplePrompts = [
      "Say your name and where you are from.",
      "Say what you are working on or studying.",
      "Say what you did earlier today.",
      "Say a completely different sentence about anything.",
      "Say one final sentence naturally."
    ];

    function showTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'tab-management') loadSpeakers();
    }

    async function checkExistingSpeakerProgress(sId) {
      if (!sId) return;
      try {
        const res = await fetch(`/api/speaker/${sId}`);
        if (res.ok) {
          const data = await res.json();
          if (data && typeof data.sample_count === 'number') {
            currentStep = Math.min(totalSteps, data.sample_count + 1);
            const stepEl = document.getElementById('enroll-step');
            if (stepEl) stepEl.innerText = currentStep;
            const promptEl = document.getElementById('enroll-prompt');
            if (promptEl) promptEl.innerText = `"${samplePrompts[(currentStep - 1) % samplePrompts.length]}"`;
            const btn = document.getElementById('btn-record-enroll');
            if (btn) btn.innerHTML = `<span>🔴 Record Sample (${currentStep}/${totalSteps})</span>`;
          }
        }
      } catch (e) {}
    }

    function syncId() {
      const name = document.getElementById('enroll-name').value.trim();
      const idInput = document.getElementById('enroll-id');
      if (name && (!idInput.value || idInput.dataset.manual !== 'true')) {
        idInput.value = name.toLowerCase().replace(/[^a-z0-9_]/g, '_');
        checkExistingSpeakerProgress(idInput.value);
      }
    }
    document.getElementById('enroll-id').addEventListener('input', function() {
      this.dataset.manual = 'true';
      checkExistingSpeakerProgress(this.value.trim());
    });
    document.getElementById('enroll-id').addEventListener('change', function() {
      checkExistingSpeakerProgress(this.value.trim());
    });

    function updateThresh(val) {
      document.getElementById('thresh-val').innerText = parseFloat(val).toFixed(2);
    }

    async function switchModel() {
      const model = document.getElementById('model-select').value;
      const res = await fetch('/api/model/switch', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `model_name=${model}`
      });
      if (res.ok) {
        alert('Active speaker encoder switched to: ' + model);
        loadSpeakers();
      }
    }

    // Pure JavaScript PCM to RIFF 16-bit Mono WAV Encoder
    function encodeWAV(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);

      function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      }

      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM format = 1
      view.setUint16(22, 1, true); // Mono channel = 1
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); // byte rate (sr * channels * 2)
      view.setUint16(32, 2, true); // block align (channels * 2)
      view.setUint16(34, 16, true); // 16 bits per sample
      writeString(view, 36, 'data');
      view.setUint32(40, samples.length * 2, true);

      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }

      return new Blob([buffer], { type: 'audio/wav' });
    }

    // Web Audio API Capture with Live Waveform Drawing and Direct WAV Encoding
    async function captureAudioWithWaveform(durationMs, canvasId, meterId) {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        }
      });
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const canvas = document.getElementById(canvasId);
      const ctx = canvas ? canvas.getContext('2d') : null;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const meterBar = document.getElementById(meterId);

      let animRunning = true;
      function draw() {
        if (!animRunning) return;
        requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);

        if (ctx && canvas) {
          ctx.fillStyle = '#0c101b';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#38bdf8';
          ctx.beginPath();

          const sliceWidth = canvas.width * 1.0 / dataArray.length;
          let x = 0;
          let sum = 0;

          for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * (canvas.height / 2);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
            sum += Math.abs(dataArray[i] - 128);
          }
          ctx.lineTo(canvas.width, canvas.height / 2);
          ctx.stroke();

          const pct = Math.min(100, (sum / dataArray.length) * 4);
          if (meterBar) meterBar.style.width = pct + '%';
        }
      }
      draw();

      const scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
      const pcmChunks = [];
      let isCapturing = true;

      scriptNode.onaudioprocess = (e) => {
        if (!isCapturing) return;
        const channelData = e.inputBuffer.getChannelData(0);
        pcmChunks.push(new Float32Array(channelData));
      };

      const muteGain = audioCtx.createGain();
      muteGain.gain.value = 0.0;
      source.connect(scriptNode);
      scriptNode.connect(muteGain);
      muteGain.connect(audioCtx.destination);

      return new Promise((resolve, reject) => {
        setTimeout(async () => {
          isCapturing = false;
          animRunning = false;
          if (meterBar) meterBar.style.width = '0%';
          try { scriptNode.disconnect(); } catch (e) {}
          try { muteGain.disconnect(); } catch (e) {}
          try { source.disconnect(); } catch (e) {}
          stream.getTracks().forEach(t => t.stop());
          const sr = audioCtx.sampleRate;
          try { await audioCtx.close(); } catch (e) {}

          const totalSamples = pcmChunks.reduce((acc, c) => acc + c.length, 0);
          if (totalSamples === 0) {
            return reject(new Error("No microphone audio samples captured. Check mic permissions."));
          }
          const merged = new Float32Array(totalSamples);
          let offset = 0;
          for (const c of pcmChunks) {
            merged.set(c, offset);
            offset += c.length;
          }
          const wavBlob = encodeWAV(merged, sr);
          resolve(wavBlob);
        }, durationMs);
      });
    }

    // 1. Identify Flow
    let checkSequence = 0;
    async function recordAndIdentify() {
      // Stop live stream if active to prevent mic contention
      if (isLiveStreaming) {
        toggleLiveStream();
      }

      checkSequence++;
      const currentSeq = checkSequence;

      const btn = document.getElementById('btn-record-ident');
      btn.classList.add('recording');
      btn.innerHTML = `<span>🎙️ Recording Check #${currentSeq} (Speak Now)...</span>`;
      document.getElementById('ident-status-badge').className = 'badge badge-red';
      document.getElementById('ident-status-badge').innerText = `Listening (#${currentSeq})...`;

      // Clear / update attribution card to active evaluating state so user knows a new check is running
      document.getElementById('ident-empty').style.display = 'none';
      const card = document.getElementById('ident-card');
      card.style.display = 'block';
      card.className = 'result-box';
      card.style.opacity = '0.7';

      const badge = document.getElementById('badge-decision');
      badge.className = 'badge badge-cyan';
      badge.innerText = `🎙️ CHECK #${currentSeq}: LISTENING...`;

      document.getElementById('res-speaker').innerText = 'Listening to speech...';
      document.getElementById('res-speaker').style.color = 'var(--text-dim)';
      document.getElementById('res-sim').innerText = 'Recording 3.0s window...';
      document.getElementById('res-conf').innerText = '...';
      document.getElementById('res-margin').innerText = '...';
      document.getElementById('res-latency').innerText = '...';

      try {
        const blob = await captureAudioWithWaveform(3000, 'canvas-ident', 'meter-ident');
        btn.classList.remove('recording');
        btn.innerHTML = '<span>⚡ Processing Neural Embedding...</span>';
        document.getElementById('ident-status-badge').className = 'badge badge-amber';
        document.getElementById('ident-status-badge').innerText = 'Classifying...';
        badge.className = 'badge badge-amber';
        badge.innerText = `⚡ CHECK #${currentSeq}: EXTRACTING EMBEDDING...`;
        document.getElementById('res-speaker').innerText = 'Comparing against centroids...';

        const formData = new FormData();
        formData.append('audio_file', blob, 'ident.wav');
        formData.append('threshold', document.getElementById('thresh-slider').value);

        const res = await fetch('/api/identify', { method: 'POST', body: formData });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: res.statusText }));
          alert('Identification error: ' + (errData.detail || 'Server error'));
          return;
        }
        const data = await res.json();
        data._seq = currentSeq;
        renderIdentificationCard(data);
      } catch (err) {
        alert('Microphone error: ' + err.message);
        card.style.opacity = '1.0';
      } finally {
        btn.classList.remove('recording');
        btn.innerHTML = '<span>🔴 Record & Identify (3.0s)</span>';
        document.getElementById('ident-status-badge').className = 'badge badge-cyan';
        document.getElementById('ident-status-badge').innerText = 'Ready';
      }
    }

    function renderIdentificationCard(data) {
      document.getElementById('ident-empty').style.display = 'none';
      const card = document.getElementById('ident-card');
      card.style.display = 'block';
      card.style.opacity = '1.0';

      const isKnown = (data.decision === 'KNOWN');
      card.className = 'result-box ' + (isKnown ? 'result-known' : 'result-unknown');

      const badge = document.getElementById('badge-decision');
      const seqStr = data._seq ? ` [Check #${data._seq}]` : '';
      badge.className = 'badge ' + (isKnown ? 'badge-green' : 'badge-amber');
      badge.innerText = (isKnown ? '✅ KNOWN SPEAKER' : '⚠️ UNKNOWN SPEAKER') + seqStr;

      document.getElementById('res-speaker').innerText = isKnown ? data.display_name : (data.decision === 'INSUFFICIENT_AUDIO' ? 'INSUFFICIENT SPEECH' : 'UNKNOWN SPEAKER');
      document.getElementById('res-speaker').style.color = isKnown ? 'var(--success)' : 'var(--warning)';
      document.getElementById('res-sim').innerText = data.similarity + ' (Threshold: ' + data.threshold + ')';
      document.getElementById('res-conf').innerText = (data.confidence * 100).toFixed(1) + '%';
      document.getElementById('res-margin').innerText = (data.margin >= 0 ? '+' : '') + data.margin;
      document.getElementById('res-latency').innerText = data.latency_ms + ' ms';

      if (data.temporal_stability) {
        document.getElementById('row-temporal').style.display = 'flex';
        document.getElementById('res-stability').innerText = (data.temporal_stability * 100).toFixed(0) + '%';
      }

      const tbody = document.querySelector('#table-alts tbody');
      tbody.innerHTML = '';
      if (data.alternatives && data.alternatives.length > 0) {
        data.alternatives.forEach(a => {
          tbody.innerHTML += `<tr><td>#${a.rank}</td><td><strong>${a.speaker_id}</strong></td><td>${a.similarity.toFixed(3)}</td></tr>`;
        });
      } else {
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; color:var(--text-dim);">No candidate scores available</td></tr>';
      }
    }

    // Toggle Live Stream Identification
    let liveTimeoutHandle = null;
    async function toggleLiveStream() {
      const btn = document.getElementById('btn-live-stream');
      const ind = document.getElementById('live-indicator');
      if (isLiveStreaming) {
        isLiveStreaming = false;
        if (liveTimeoutHandle) {
          clearTimeout(liveTimeoutHandle);
          liveTimeoutHandle = null;
        }
        btn.innerHTML = '<span>⚡ Live Stream</span>';
        btn.className = 'btn btn-secondary';
        if (ind) ind.style.display = 'none';
      } else {
        isLiveStreaming = true;
        btn.innerHTML = '<span>⏹️ Stop Live</span>';
        btn.className = 'btn btn-danger';
        if (ind) ind.style.display = 'inline-flex';
        liveLoop();
      }
    }

    async function liveLoop() {
      if (!isLiveStreaming) return;
      try {
        const blob = await captureAudioWithWaveform(1600, 'canvas-ident', 'meter-ident');
        if (!isLiveStreaming) return;
        const formData = new FormData();
        formData.append('audio_file', blob, 'stream.wav');
        formData.append('threshold', document.getElementById('thresh-slider').value);

        const res = await fetch('/api/speaker/live/identify', { method: 'POST', body: formData });
        if (res.ok && isLiveStreaming) {
          const data = await res.json();
          renderIdentificationCard(data);
        }
      } catch (e) {
        console.error('Live stream error:', e);
      }
      if (isLiveStreaming) {
        liveTimeoutHandle = setTimeout(liveLoop, 200);
      }
    }

    // 2. Enrollment Flow
    async function recordEnrollmentSample() {
      const id = document.getElementById('enroll-id').value.trim();
      const name = document.getElementById('enroll-name').value.trim() || id;
      if (!id) return alert('Please enter a person name.');

      const btn = document.getElementById('btn-record-enroll');
      btn.classList.add('recording');
      btn.innerHTML = `<span>🎙️ Recording Sample ${currentStep}/${totalSteps}...</span>`;

      try {
        const blob = await captureAudioWithWaveform(3000, 'canvas-enroll', 'meter-enroll');
        btn.classList.remove('recording');
        btn.innerHTML = '<span>⚡ Verifying Acoustic Quality...</span>';

        const formData = new FormData();
        formData.append('speaker_id', id);
        formData.append('display_name', name);
        formData.append('audio_file', blob, 'sample.wav');

        const res = await fetch('/api/enroll', { method: 'POST', body: formData });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: res.statusText }));
          alert('Enrollment error: ' + (errData.detail || 'Server error'));
          return;
        }
        const data = await res.json();

        const qCard = document.getElementById('enroll-quality-card');
        qCard.style.display = 'block';

        const vBadge = document.getElementById('quality-verdict-badge');
        const chips = document.getElementById('quality-chips');

        if (data.success) {
          vBadge.className = 'badge badge-green';
          vBadge.innerText = '✓ ACCEPTED & STORED';

          chips.innerHTML = `
            <span class="quality-chip chip-ok">Speech: ${data.quality.speech_duration_sec}s</span>
            <span class="quality-chip chip-ok">RMS: ${data.quality.rms}</span>
            <span class="quality-chip chip-ok">SNR: ${data.quality.snr_db} dB</span>
            <span class="quality-chip chip-ok">Saved: ${data.profile.last_sample_file}</span>
          `;

          if (data.profile && typeof data.profile.sample_count === 'number') {
            currentStep = data.profile.sample_count + 1;
          } else {
            currentStep++;
          }
          if (currentStep > totalSteps) {
            alert(`🎉 Success! ${name} is now enrolled with ${totalSteps} saved utterances.\nProfile stored locally.`);
            currentStep = 1;
            loadSpeakers();
          }
        } else {
          vBadge.className = 'badge badge-red';
          vBadge.innerText = '✗ SAMPLE REJECTED';

          const reason = data.quality && data.quality.reason ? data.quality.reason : 'Failed quality checks';
          const dur = data.quality ? data.quality.duration_sec : 0;
          const snr = data.quality ? data.quality.snr_db : 0;
          chips.innerHTML = `
            <span class="quality-chip chip-fail">Reason: ${reason}</span>
            <span class="quality-chip chip-warn">Duration: ${dur}s</span>
            <span class="quality-chip chip-warn">SNR: ${snr} dB</span>
          `;
        }

        const stepEl = document.getElementById('enroll-step');
        if (stepEl) stepEl.innerText = currentStep;
        const promptEl = document.getElementById('enroll-prompt');
        if (promptEl) promptEl.innerText = `"${samplePrompts[(currentStep - 1) % samplePrompts.length]}"`;
      } catch (err) {
        alert('Microphone recording error: ' + err.message);
      } finally {
        btn.classList.remove('recording');
        btn.innerHTML = `<span>🔴 Record Sample (${currentStep}/${totalSteps})</span>`;
      }
    }

    // 3. Speaker Management Flow
    async function loadSpeakers() {
      try {
        const res = await fetch('/api/speakers');
        const data = await res.json();
        document.getElementById('db-status').innerText = 'Active Profiles: ' + data.speakers.length;
        document.getElementById('tab-count').innerText = data.speakers.length;
        document.getElementById('storage-dir-display').innerText = data.storage_path;

        const tbody = document.querySelector('#table-speakers-mgmt tbody');
        tbody.innerHTML = '';
        if (data.speakers.length === 0) {
          document.getElementById('mgmt-empty').style.display = 'block';
        } else {
          document.getElementById('mgmt-empty').style.display = 'none';
          data.speakers.forEach(s => {
            tbody.innerHTML += `
              <tr>
                <td><strong>${s.display_name}</strong></td>
                <td><code style="color:var(--text-dim);">${s.speaker_id}</code></td>
                <td>${s.sample_count} samples</td>
                <td><span class="badge badge-cyan">${s.model_name}</span></td>
                <td><span class="badge badge-green">${s.recordings_count} .wav files</span></td>
                <td>
                  <button class="btn btn-secondary" style="width:auto; padding:4px 10px; font-size:12px; display:inline-flex;" onclick="addSampleToSpeaker('${s.speaker_id}', '${s.display_name}')">➕ Add Sample</button>
                  <button class="btn btn-danger" onclick="deleteSpeaker('${s.speaker_id}', '${s.display_name}')">🗑️ Delete</button>
                </td>
              </tr>
            `;
          });
        }
      } catch (e) {
        console.error('Failed to load speakers:', e);
      }
    }

    function addSampleToSpeaker(id, name) {
      document.getElementById('enroll-id').value = id;
      document.getElementById('enroll-name').value = name;
      showTab('tab-enroll');
    }

    async function deleteSpeaker(id, name) {
      if (!confirm(`Are you sure you want to delete '${name}'?\nThis permanently removes the profile and all saved local recordings.`)) return;
      const res = await fetch(`/api/speaker/${id}`, { method: 'DELETE' });
      if (res.ok) {
        loadSpeakers();
      } else {
        alert('Failed to delete speaker.');
      }
    }

    // 4. Blind Test Flow
    async function recordBlindTestTrial() {
      const trueSpk = document.getElementById('blind-true').value.trim();
      if (!trueSpk) return alert('Evaluator: please enter Ground Truth Name (or "unknown").');
      const cond = document.getElementById('blind-cond').value;

      const blob = await captureAudioWithWaveform(3000, 'canvas-ident', 'meter-ident');
      const formData = new FormData();
      formData.append('true_speaker', trueSpk);
      formData.append('condition', cond);
      formData.append('audio_file', blob, 'blind.wav');

      const res = await fetch('/api/blind-test/record', { method: 'POST', body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        alert('Blind test error: ' + (errData.detail || 'Server error'));
        return;
      }
      const data = await res.json();
      alert(`Blind Trial #${data.test_id} recorded. (Result hidden in blind test registry)`);
      document.getElementById('blind-true').value = '';
    }

    async function loadBlindSummary() {
      const res = await fetch('/api/blind-test/summary');
      const data = await res.json();
      document.getElementById('blind-scorecard').style.display = 'block';
      document.getElementById('sum-known-acc').innerText = data.known_accuracy_percent + '%';
      document.getElementById('sum-unk-rej').innerText = data.unknown_rejection_percent + '%';
      document.getElementById('sum-all-acc').innerText = data.overall_accuracy_percent + '%';
      document.getElementById('sum-lat').innerText = data.mean_latency_ms + ' ms';

      const tbody = document.querySelector('#table-blind-trials tbody');
      tbody.innerHTML = '';
      data.test_cases.forEach(t => {
        const vBadge = t.is_correct ? '<span style="color:var(--success); font-weight:700;">CORRECT</span>' : '<span style="color:var(--danger); font-weight:700;">WRONG</span>';
        tbody.innerHTML += `<tr><td>#${t.test_id}</td><td>${t.true_speaker}</td><td><strong>${t.predicted_speaker}</strong></td><td>${t.decision_status}</td><td>${t.similarity.toFixed(3)}</td><td>${(t.confidence*100).toFixed(0)}%</td><td>${t.condition}</td><td>${vBadge}</td></tr>`;
      });
    }

    async function resetBlindTest() {
      if (confirm('Reset blind evaluation trials?')) {
        await fetch('/api/blind-test/reset', { method: 'POST' });
        document.getElementById('blind-scorecard').style.display = 'none';
      }
    }

    // Initialize
    loadSpeakers();
  </script>
</body>
</html>
"""


def launch_web_demo(host: str = "0.0.0.0", port: int = 8050, model_name: str = "campp", db_dir: str = "demo_data/speakers"):
    """Run interactive web demonstration server."""
    app = create_demo_app(db_dir=db_dir, model_name=model_name)
    print(f"\n============================================================")
    print(f" EchoLocate — Persistent Speaker Attribution System")
    print(f" Local Browser URL:    http://localhost:{port}")
    print(f" Persistent Data Dir:  {Path(db_dir).resolve()}")
    print(f"============================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
