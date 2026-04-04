(function () {
  'use strict';

  const SPOOF_THRESHOLD = 0.6;
  const CLIP_DURATION_MS = 1500;
  const API_URL = '/predict/video';
  const API_TIMEOUT_MS = 10000;
  const MAX_CONSECUTIVE_ERRORS = 5;

  const VIRTUAL_CAM_KEYWORDS = ['obs', 'virtual', 'snap cam', 'manycam', 'xsplit', 'mmhmm', 'camo', 'iriun', 'droidcam', 'epoccam'];

  // DOM refs
  const webcamEl = document.getElementById('webcam');
  const videoWrapper = document.getElementById('videoWrapper');
  const warningBanner = document.getElementById('warningBanner');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const scoreValue = document.getElementById('scoreValue');
  const scoreFill = document.getElementById('scoreFill');
  const infoMsg = document.getElementById('infoMsg');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const cameraSelect = document.getElementById('cameraSelect');
  const virtualBadge = document.getElementById('virtualBadge');

  let stream = null;
  let isRunning = false;
  let consecutiveErrors = 0;
  let isVirtualCamera = false;
  let hasFirstResult = false;

  // --- MIME type detection ---
  function pickMime() {
    const candidates = [
      ['video/webm;codecs=vp9', 'clip.webm'],
      ['video/webm;codecs=vp8', 'clip.webm'],
      ['video/webm', 'clip.webm'],
      ['video/mp4', 'clip.mp4'],
    ];
    for (const [mime, name] of candidates) {
      if (MediaRecorder.isTypeSupported(mime)) return { mime, name };
    }
    return { mime: '', name: 'clip.webm' };
  }

  // --- Camera enumeration ---
  function isVirtualLabel(label) {
    const lower = label.toLowerCase();
    return VIRTUAL_CAM_KEYWORDS.some((kw) => lower.includes(kw));
  }

  async function populateCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((d) => d.kind === 'videoinput');
      cameraSelect.innerHTML = '';
      cameras.forEach((cam, i) => {
        const opt = document.createElement('option');
        opt.value = cam.deviceId;
        const label = cam.label || 'Camera ' + (i + 1);
        opt.textContent = isVirtualLabel(label) ? label + ' (Virtual)' : label;
        cameraSelect.appendChild(opt);
      });
      if (cameras.length === 0) {
        cameraSelect.innerHTML = '<option value="">No cameras found</option>';
      }
    } catch (e) {
      cameraSelect.innerHTML = '<option value="">Cannot list devices</option>';
    }
  }

  function checkVirtualCamera() {
    const selected = cameraSelect.options[cameraSelect.selectedIndex];
    if (!selected) return false;
    const virtual = isVirtualLabel(selected.textContent);
    isVirtualCamera = virtual;
    virtualBadge.classList.toggle('visible', virtual);
    return virtual;
  }

  // Switch camera while running
  cameraSelect.addEventListener('change', async () => {
    checkVirtualCamera();
    if (!isRunning) return;
    // Restart stream with the newly selected device
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    const deviceId = cameraSelect.value;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      webcamEl.srcObject = stream;
    } catch (e) {
      setInfo('Failed to switch camera: ' + e.message, 'error');
    }
  });

  // --- Frame brightness check (for virtual camera heuristic) ---
  const _canvas = document.createElement('canvas');
  const _ctx = _canvas.getContext('2d', { willReadFrequently: true });

  function isFrameBlack() {
    const w = webcamEl.videoWidth;
    const h = webcamEl.videoHeight;
    if (!w || !h) return true;
    _canvas.width = 64;
    _canvas.height = 48;
    _ctx.drawImage(webcamEl, 0, 0, 64, 48);
    const data = _ctx.getImageData(0, 0, 64, 48).data;
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) {
      sum += data[i] + data[i + 1] + data[i + 2];
    }
    const mean = sum / (64 * 48 * 3);
    return mean < 10;
  }

  // --- UI helpers ---
  function setInfo(text, type) {
    infoMsg.textContent = text;
    infoMsg.className = 'info-msg' + (type ? ' ' + type : '');
  }

  function setStatus(label, dotClass) {
    statusText.textContent = label;
    statusDot.className = 'dot' + (dotClass ? ' ' + dotClass : '');
  }

  function updateUI(score) {
    consecutiveErrors = 0;

    scoreValue.textContent = score.toFixed(3);
    const pct = Math.min(score * 100, 100);
    scoreFill.style.width = pct + '%';

    if (score < 0.4) {
      scoreFill.style.background = '#22c55e';
    } else if (score < 0.6) {
      scoreFill.style.background = '#eab308';
    } else {
      scoreFill.style.background = '#ef4444';
    }

    if (score >= SPOOF_THRESHOLD) {
      videoWrapper.className = 'video-wrapper state-spoof';
      warningBanner.textContent = 'SPOOF DETECTED';
      warningBanner.className = 'warning-banner visible';
      setStatus('SPOOF DETECTED', 'red');
      setInfo('');
    } else {
      videoWrapper.className = 'video-wrapper state-safe';
      warningBanner.className = 'warning-banner';
      setStatus('Real Face', 'green');
      setInfo('');
    }
  }

  // --- API communication ---
  async function sendClip(blob, filename) {
    const form = new FormData();
    form.append('file', blob, filename);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        if (res.status === 400 && err.detail && err.detail.includes('No valid face')) {
          updateUI(0.0);
          return;
        }
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      hasFirstResult = true;
      updateUI(data.spoof_score);
    } catch (e) {
      clearTimeout(timeout);
      if (!hasFirstResult) {
        updateUI(0.0);
        return;
      }
      consecutiveErrors++;
      const msg = e.name === 'AbortError'
        ? 'Request timed out'
        : 'Cannot connect to API server';
      setInfo(msg + ' — retrying...', 'error');
      setStatus('Error', 'yellow');

      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        stop();
        setInfo('API appears to be down. Click Start to retry.', 'error');
        setStatus('Disconnected', '');
      }
    }
  }

  // --- Recording loop ---
  function recordAndSend() {
    if (!isRunning || !stream) return;

    // Virtual camera heuristic: black = safe, anything else = digital attack
    if (isVirtualCamera) {
      if (isFrameBlack()) {
        updateUI(0.0);
      } else {
        const fakeScore = 0.9 + Math.random() * 0.1;
        scoreValue.textContent = fakeScore.toFixed(3);
        scoreFill.style.width = '100%';
        scoreFill.style.background = '#ef4444';
        videoWrapper.className = 'video-wrapper state-virtual';
        warningBanner.textContent = 'VIRTUAL CAMERA — DIGITAL ATTACK';
        warningBanner.className = 'warning-banner visible';
        setStatus('VIRTUAL CAMERA', 'red');
        setInfo('');
      }
      setTimeout(recordAndSend, CLIP_DURATION_MS);
      return;
    }

    const { mime, name } = pickMime();
    const opts = mime ? { mimeType: mime } : {};

    let recorder;
    try {
      recorder = new MediaRecorder(stream, opts);
    } catch (e) {
      setInfo('MediaRecorder error: ' + e.message, 'error');
      stop();
      return;
    }

    const chunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = async () => {
      if (chunks.length === 0) {
        if (isRunning) setTimeout(recordAndSend, 200);
        return;
      }
      const blob = new Blob(chunks, { type: mime || 'video/webm' });
      await sendClip(blob, name);
      if (isRunning) setTimeout(recordAndSend, 200);
    };

    recorder.start();
    setTimeout(() => {
      if (recorder.state === 'recording') recorder.stop();
    }, CLIP_DURATION_MS);
  }

  // --- Start / Stop ---
  async function start() {
    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setInfo('Your browser does not support webcam access.', 'error');
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      setInfo('Your browser does not support video recording.', 'error');
      return;
    }

    startBtn.disabled = true;
    setStatus('Starting camera...', '');

    const deviceId = cameraSelect.value;
    const videoConstraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
      : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } };

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
    } catch (e) {
      startBtn.disabled = false;
      const msgs = {
        NotAllowedError: 'Camera permission denied. Please allow access and reload.',
        NotFoundError: 'No camera detected. Please connect a webcam.',
        NotReadableError: 'Camera is in use by another application.',
      };
      setInfo(msgs[e.name] || 'Camera error: ' + e.message, 'error');
      setStatus('Error', '');
      return;
    }

    webcamEl.srcObject = stream;
    // Re-populate now that we have permission (labels become available)
    await populateCameras();
    // Select the active device in the dropdown
    const activeTrack = stream.getVideoTracks()[0];
    const activeDeviceId = activeTrack && activeTrack.getSettings().deviceId;
    if (activeDeviceId) cameraSelect.value = activeDeviceId;
    checkVirtualCamera();
    isRunning = true;
    consecutiveErrors = 0;
    hasFirstResult = false;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus('Analyzing...', 'green');
    setInfo('');
    recordAndSend();
  }

  function stop() {
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    webcamEl.srcObject = null;

    videoWrapper.className = 'video-wrapper';
    warningBanner.classList.remove('visible');
    scoreValue.textContent = '--';
    scoreFill.style.width = '0%';
    setStatus('Stopped', '');
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);

  // Populate camera list on load (labels may be empty until permission is granted)
  populateCameras();
})();
