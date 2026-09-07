"""Web GUI and Live Streaming Server for Portal-BOT."""

import asyncio
from dataclasses import asdict
import json
import logging
import os
import time
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from portal_bot.config import BotConfig, DEFAULT_CONFIG
from portal_bot.core.bot_engine import BotEngine
from portal_bot.core.events import event_bus
from portal_bot.core.types import BotStatus
from portal_bot.engine_bridge.portal_config import install_portal_cfg
from portal_bot.utils.logger import bot_log

logger = logging.getLogger("portal_bot.web")


def create_app(engine: BotEngine) -> FastAPI:
    app = FastAPI(title="Portal-BOT Control Center")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    active_websockets: List[WebSocket] = []

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        return HTMLResponse("<h1>Portal-BOT Web Dashboard</h1>")

    @app.get("/api/status")
    async def get_status():
        state = engine.get_state_snapshot()
        return JSONResponse({
            "status": engine.status.value,
            "fps": round(state.fps, 1) if state else 0.0,
            "confidence": state.confidence if state else 1.0,
            "exit_door_open": state.exit_door_open if state else False,
            "button_pressed": state.button_pressed if state else False,
            "blue_portal": state.portals.blue_active if state else False,
            "orange_portal": state.portals.orange_active if state else False,
            "gun_state": state.player.gun_state.value if state else "No Gun",
            "holding_cube": state.player.holding_cube if state else False,
            "objects_count": len(state.objects) if state else 0,
            "current_goal": engine.agent.current_goal.reason if engine.agent.current_goal else "Idle",
            "active_subgoal_type": engine.agent.current_goal.goal_type.value if engine.agent.current_goal else "None",
            "engine_log_connected": engine.console_reader.state.connected,
            "chamber_id": state.current_chamber if state else 0,
            "player_yaw": round(state.player.yaw, 1) if state else 0.0,
            "player_pitch": round(state.player.pitch, 1) if state else 0.0,
        })

    @app.post("/api/start")
    async def start_bot():
        engine.start()
        return {"status": engine.status.value, "message": "Bot started"}

    @app.post("/api/pause")
    async def pause_bot():
        engine.pause()
        return {"status": engine.status.value, "message": "Bot paused"}

    @app.post("/api/resume")
    async def resume_bot():
        engine.resume()
        return {"status": engine.status.value, "message": "Bot resumed"}

    @app.post("/api/stop")
    async def stop_bot():
        engine.stop()
        return {"status": engine.status.value, "message": "Bot stopped"}

    @app.post("/api/emergency_stop")
    async def emergency_stop():
        engine.emergency_stop()
        return {"status": engine.status.value, "message": "Emergency safety stop executed"}

    @app.post("/api/install_cfg")
    async def api_install_cfg():
        success = install_portal_cfg()
        return {"success": success, "message": "Portal configuration installed successfully" if success else "Failed installing config"}

    @app.get("/api/logs")
    async def get_logs(limit: int = 50):
        return {"logs": bot_log.get_recent_logs(limit)}

    def generate_video_stream():
        """Fast MJPEG stream from debug frames with live preview support."""
        target_fps = max(10, engine.config.ui.stream_fps)
        target_delay = 1.0 / target_fps

        while True:
            t0 = time.time()
            try:
                debug_frame = engine.get_debug_frame()
                if debug_frame is None:
                    debug_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        debug_frame, "APERTURE VISION: WAITING FOR SIGNAL", (110, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2
                    )

                ret, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, engine.config.ui.jpeg_quality])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
            except Exception as e:
                logger.debug(f"Video stream frame error: {e}")

            elapsed = time.time() - t0
            time.sleep(max(0.005, target_delay - elapsed))

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            generate_video_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        active_websockets.append(websocket)
        try:
            while True:
                state = engine.get_state_snapshot()
                payload = {
                    "status": engine.status.value,
                    "fps": round(state.fps, 1) if state else 60.0,
                    "confidence": round(state.confidence, 2) if state else 1.0,
                    "door_open": state.exit_door_open if state else False,
                    "button_pressed": state.button_pressed if state else False,
                    "blue_portal": state.portals.blue_active if state else False,
                    "orange_portal": state.portals.orange_active if state else False,
                    "gun_state": state.player.gun_state.value if state else "No Gun",
                    "holding_cube": state.player.holding_cube if state else False,
                    "engine_log_connected": engine.console_reader.state.connected,
                    "chamber_id": state.current_chamber if state else 0,
                    "goal_type": engine.agent.current_goal.goal_type.value if engine.agent.current_goal else "Idle",
                    "goal_reason": engine.agent.current_goal.reason if engine.agent.current_goal else "Waiting for start",
                    "yaw": round(state.player.yaw, 1) if state else 0.0,
                    "pitch": round(state.player.pitch, 1) if state else 0.0,
                    "recent_logs": bot_log.get_recent_logs(8)
                }
                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            if websocket in active_websockets:
                active_websockets.remove(websocket)
        except Exception:
            if websocket in active_websockets:
                active_websockets.remove(websocket)

    return app
