// Portal-BOT Web Dashboard JavaScript v2.0

document.addEventListener('DOMContentLoaded', () => {
    // UI Element References
    const botStatusPill = document.getElementById('botStatusPill');
    const statusDot = document.getElementById('statusDot');
    const botStatusText = document.getElementById('botStatusText');
    const engineDot = document.getElementById('engineDot');
    const engineLogText = document.getElementById('engineLogText');
    
    const fpsBadge = document.getElementById('fpsBadge');
    const gunBadge = document.getElementById('gunBadge');
    const confBadge = document.getElementById('confBadge');
    
    const goalTypeBadge = document.getElementById('goalTypeBadge');
    const goalText = document.getElementById('goalText');
    const treeMainGoal = document.getElementById('treeMainGoal');
    const treeSubgoal = document.getElementById('treeSubgoal');
    
    // Telemetry
    const teleGunState = document.getElementById('teleGunState');
    const teleHand = document.getElementById('teleHand');
    const teleBluePortal = document.getElementById('teleBluePortal');
    const teleOrangePortal = document.getElementById('teleOrangePortal');
    const teleButton = document.getElementById('teleButton');
    const teleDoor = document.getElementById('teleDoor');
    
    // Logs
    const logStream = document.getElementById('logStream');
    const btnClearLogs = document.getElementById('btnClearLogs');

    // Controls
    const btnStart = document.getElementById('btnStart');
    const btnPause = document.getElementById('btnPause');
    const btnResume = document.getElementById('btnResume');
    const btnStop = document.getElementById('btnStop');
    const btnEmergencyStop = document.getElementById('btnEmergencyStop');
    const btnInstallCfg = document.getElementById('btnInstallCfg');

    let isRunning = false;
    let isPaused = false;

    async function apiPost(endpoint) {
        try {
            const res = await fetch(`/api/${endpoint}`, { method: 'POST' });
            return await res.json();
        } catch (err) {
            console.error(`API Error on ${endpoint}:`, err);
        }
    }

    btnStart.addEventListener('click', async () => {
        await apiPost('start');
        setRunningState(true, false);
    });

    btnPause.addEventListener('click', async () => {
        await apiPost('pause');
        setRunningState(true, true);
    });

    btnResume.addEventListener('click', async () => {
        await apiPost('resume');
        setRunningState(true, false);
    });

    btnStop.addEventListener('click', async () => {
        await apiPost('stop');
        setRunningState(false, false);
    });

    btnEmergencyStop.addEventListener('click', async () => {
        await apiPost('emergency_stop');
        setRunningState(false, false);
    });

    btnInstallCfg.addEventListener('click', async () => {
        const res = await apiPost('install_cfg');
        if (res && res.message) {
            alert(res.message);
        }
    });

    btnClearLogs.addEventListener('click', () => {
        logStream.innerHTML = '';
    });

    // Emergency Stop Key (F8)
    window.addEventListener('keydown', (e) => {
        if (e.key === 'F8') {
            e.preventDefault();
            apiPost('emergency_stop');
            setRunningState(false, false);
        }
    });

    function setRunningState(running, paused) {
        isRunning = running;
        isPaused = paused;

        if (running && !paused) {
            btnStart.style.display = 'none';
            btnPause.style.display = 'inline-flex';
            btnResume.style.display = 'none';
            btnStop.style.display = 'inline-flex';
        } else if (running && paused) {
            btnStart.style.display = 'none';
            btnPause.style.display = 'none';
            btnResume.style.display = 'inline-flex';
            btnStop.style.display = 'inline-flex';
        } else {
            btnStart.style.display = 'inline-flex';
            btnPause.style.display = 'inline-flex';
            btnResume.style.display = 'none';
            btnStop.style.display = 'inline-flex';
        }
    }

    function updateStatusUI(status) {
        botStatusText.textContent = status.toUpperCase();
        statusDot.className = 'status-indicator';

        if (status === 'Running' || status === 'Solving Puzzle') {
            statusDot.classList.add('running');
            setRunningState(true, false);
        } else if (status === 'Paused') {
            statusDot.classList.add('paused');
            setRunningState(true, true);
        } else if (status.includes('Stuck') || status.includes('Dead') || status === 'Error') {
            statusDot.classList.add('stuck');
        } else {
            setRunningState(false, false);
        }
    }

    let ws = null;
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('Telemetry WebSocket connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Status
                updateStatusUI(data.status);
                
                // Engine Log Bridge Indicator
                if (data.engine_log_connected) {
                    engineDot.className = 'engine-dot connected';
                    engineLogText.textContent = `GAME LOG: LIVE (CH 0${data.chamber_id})`;
                } else {
                    engineDot.className = 'engine-dot';
                    engineLogText.textContent = 'GAME LOG: DISCONNECTED';
                }

                // Badges
                fpsBadge.textContent = `FPS: ${data.fps}`;
                gunBadge.textContent = `GUN: ${data.gun_state.toUpperCase()}`;
                confBadge.textContent = `CONF: ${data.confidence}`;
                
                // Goals
                goalTypeBadge.textContent = data.goal_type;
                goalText.textContent = data.goal_reason;
                treeMainGoal.textContent = `1. Main Goal: Complete Chamber 0${data.chamber_id}`;
                treeSubgoal.textContent = `${data.goal_type}: ${data.goal_reason}`;

                // Telemetry
                teleGunState.textContent = data.gun_state.toUpperCase();
                teleHand.textContent = data.holding_cube ? 'CUBE (IN HAND)' : 'NONE';
                teleHand.style.color = data.holding_cube ? '#00d2ff' : '#8b9bb4';

                teleBluePortal.textContent = data.blue_portal ? 'ACTIVE' : 'INACTIVE';
                teleBluePortal.style.color = data.blue_portal ? '#00d2ff' : '#8b9bb4';

                teleOrangePortal.textContent = data.orange_portal ? 'ACTIVE' : 'INACTIVE';
                teleOrangePortal.style.color = data.orange_portal ? '#ff8800' : '#8b9bb4';

                teleButton.textContent = data.button_pressed ? 'PRESSED' : 'UNPRESSED';
                teleButton.style.color = data.button_pressed ? '#10b981' : '#ef4444';

                teleDoor.textContent = data.door_open ? 'OPEN' : 'CLOSED';
                teleDoor.style.color = data.door_open ? '#10b981' : '#8b9bb4';

                // Update Logs
                if (data.recent_logs && data.recent_logs.length > 0) {
                    renderLogs(data.recent_logs);
                }

            } catch (err) {
                console.error('WebSocket parse error:', err);
            }
        };

        ws.onclose = () => {
            setTimeout(connectWebSocket, 1500);
        };
    }

    const seenLogMessages = new Set();

    function renderLogs(logs) {
        let appended = false;
        logs.forEach(item => {
            const key = `${item.time}_${item.category}_${item.message}`;
            if (!seenLogMessages.has(key)) {
                seenLogMessages.add(key);
                if (seenLogMessages.size > 200) {
                    const firstKey = seenLogMessages.values().next().value;
                    seenLogMessages.delete(firstKey);
                }

                const row = document.createElement('div');
                const cat = (item.category || '').toLowerCase();
                row.className = `log-row ${cat}`;
                row.innerHTML = `<span class="log-time">[${item.time}]</span> ${item.category ? `<strong>${item.category}:</strong> ` : ''}${item.message}`;
                logStream.appendChild(row);
                appended = true;
            }
        });

        if (appended) {
            logStream.scrollTop = logStream.scrollHeight;
        }
    }

    connectWebSocket();
});
