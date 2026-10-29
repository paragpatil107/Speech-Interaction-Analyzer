(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    model: $('modelSelect'), device: $('deviceSelect'), token: $('hfToken'),
    start: $('startRecord'), stop: $('stopRecord'), analyze: $('analyzeSelected'), close: $('closeApp'),
    status: $('statusLine'), time: $('recordingTime'), progressTrack: $('progressTrack'), progressBar: $('progressBar'),
    sessions: $('sessionList'), refresh: $('refreshSessions'), subtitle: $('resultsSubheading'),
    empty: $('emptyResults'), ready: $('readyResults'), results: $('reportResults'), links: $('downloadLinks'),
    duration: $('statDuration'), speakers: $('statSpeakers'), overlap: $('statOverlap'), overlapPercent: $('statOverlapPercent'),
    pauses: $('statPauses'), timeline: $('timelineImage'), transcript: $('transcriptText'), wer: $('werText'),
    overlapSummary: $('overlapSummary'), overlapEvents: $('overlapEvents'), turns: $('speakerTurnsBody'),
    downloadReport: $('downloadReport'), downloadCsv: $('downloadCsv'), copy: $('copyTranscript')
  };
  let selectedSession = null;
  let currentStatus = null;
  let sessionSignature = '';
  let lastAutoLoaded = '';
  let loadedReport = null;
  let localErrorUntil = 0;

  const fmtDuration = (seconds) => {
    if (seconds == null || !Number.isFinite(Number(seconds))) return '—';
    const total = Math.max(0, Math.round(Number(seconds)));
    return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
  };
  const fixed = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const settings = () => {
    const device = els.device.value;
    return { model: els.model.value, device, compute_type: device === 'cuda' ? 'float16' : 'int8', min_overlap: 0.15, sample_rate: 16000 };
  };
  const tokenValue = () => els.token.value.trim();

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }
  async function postJson(url, data) {
    return requestJson(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
  }
  function setStatus(message, isError = false) {
    els.status.textContent = message || '';
    els.status.classList.toggle('error', isError);
    if (isError) localErrorUntil = Date.now() + 4000;
  }
  function getBusy() { return !!currentStatus && (currentStatus.recording || currentStatus.analyzing); }
  function refreshButtons() {
    const busy = getBusy();
    els.start.disabled = busy;
    els.stop.disabled = !(currentStatus && currentStatus.recording);
    const selected = currentStatus?.sessions?.find((s) => s.id === selectedSession);
    els.analyze.disabled = busy || !selected || !['ready', 'complete', 'error'].includes(selected.status);
    els.analyze.textContent = selected?.status === 'complete' ? 'Analyze again' : 'Analyze';
    els.time.hidden = !(currentStatus && currentStatus.recording);
    els.progressTrack.hidden = !(currentStatus && currentStatus.analyzing);
    if (currentStatus?.analyzing) els.progressBar.style.width = `${Math.max(0, Math.min(100, currentStatus.progress || 0))}%`;
  }

  function renderSessions(sessions) {
    const signature = JSON.stringify(sessions.map((s) => [s.id, s.status, s.created, s.duration_seconds, s.speaker_count, s.overlap_seconds]));
    if (signature === sessionSignature) return;
    sessionSignature = signature;
    if (!sessions.length) {
      els.sessions.innerHTML = '<p class="muted">No recordings yet.</p>';
      return;
    }
    els.sessions.innerHTML = sessions.map((s) => {
      const duration = s.duration_seconds == null ? '' : `${fixed(s.duration_seconds, 1)} sec`;
      const info = [s.created, duration, s.speaker_count == null ? '' : `${s.speaker_count} speaker(s)`].filter(Boolean).join(' · ');
      return `<button type="button" class="session-row ${s.id === selectedSession ? 'selected' : ''}" data-session="${esc(s.id)}">
        <span class="session-top"><span class="session-name">${esc(s.id)}</span><span class="session-status">${esc(s.status)}</span></span>
        <span class="session-meta">${esc(info || 'Recording')}</span>
      </button>`;
    }).join('');
    els.sessions.querySelectorAll('[data-session]').forEach((button) => button.addEventListener('click', () => selectSession(button.dataset.session)));
  }

  function hideResults() {
    els.empty.hidden = true;
    els.ready.hidden = true;
    els.results.hidden = true;
    els.links.hidden = true;
  }
  function clearResult() {
    loadedReport = null;
    els.empty.hidden = false;
    els.ready.hidden = true;
    els.results.hidden = true;
    els.links.hidden = true;
    els.subtitle.textContent = 'Select a previous recording or record a new one.';
  }
  async function selectSession(sessionId) {
    selectedSession = sessionId;
    sessionSignature = '';
    if (currentStatus) renderSessions(currentStatus.sessions || []);
    await loadSession(sessionId);
    refreshButtons();
  }

  async function loadSession(sessionId) {
    if (!sessionId) { clearResult(); return; }
    try {
      const data = await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`);
      els.subtitle.textContent = `${data.id}${data.created ? ` · ${data.created}` : ''}`;
      if (!data.report) {
        loadedReport = null;
        els.empty.hidden = true;
        els.ready.hidden = false;
        els.results.hidden = true;
        els.links.hidden = true;
        return;
      }
      showReport(data.id, data.report);
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  function showReport(sessionId, report) {
    loadedReport = report;
    const audio = report.audio || {};
    const overlap = report.overlap || {};
    const pauses = report.pauses || {};
    const transcript = report.transcription?.text || '';
    els.empty.hidden = true;
    els.ready.hidden = true;
    els.results.hidden = false;
    els.links.hidden = false;
    els.subtitle.textContent = `${sessionId} · ${report.models?.asr || 'Transcription model not recorded'} · ${report.models?.device || 'device unknown'}`;
    els.duration.textContent = `${fixed(audio.duration_seconds, 1)} sec`;
    els.speakers.textContent = report.speaker_count ?? '—';
    els.overlap.textContent = `${fixed(overlap.total_seconds, 2)} sec`;
    els.overlapPercent.textContent = `${fixed(Number(overlap.recording_ratio ?? 0) * 100, 2)}% of recording`;
    els.pauses.textContent = pauses.count ?? '—';
    els.timeline.onerror = () => { els.timeline.hidden = true; };
    els.timeline.onload = () => { els.timeline.hidden = false; };
    els.timeline.hidden = false;
    els.timeline.src = `/api/sessions/${encodeURIComponent(sessionId)}/files/timeline.png?t=${Date.now()}`;
    els.transcript.textContent = transcript || '(No transcript produced.)';
    const evalData = report.recognition_evaluation;
    els.wer.textContent = evalData ? `Whole-recording WER: ${fixed(evalData.wer * 100, 2)}% (${evalData.errors} edits / ${evalData.reference_words} reference words)` : 'WER not calculated — no reference transcript was supplied.';
    const events = report.candidate_interruption_events || [];
    els.overlapSummary.textContent = `${overlap.interval_count || 0} detected overlap interval(s); minimum threshold ${fixed(0.15, 2)} sec.`;
    els.overlapEvents.innerHTML = events.length ? events.map((event, index) => `<div class="event">Event ${index + 1}: ${fixed(event.start, 2)}–${fixed(event.end, 2)} sec · duration ${fixed(event.duration ?? (event.end - event.start), 2)} sec</div>`).join('') : '<p class="muted">No overlap candidates were reported.</p>';
    const turns = report.speaker_turns || [];
    els.turns.innerHTML = turns.length ? turns.map((turn) => `<tr><td>${esc(turn.speaker || 'Unknown')}</td><td>${fixed(turn.start, 2)}</td><td>${fixed(turn.end, 2)}</td><td>${fixed(turn.duration ?? (turn.end - turn.start), 2)}</td></tr>`).join('') : '<tr><td colspan="4">No speaker turns available.</td></tr>';
    const base = `/api/sessions/${encodeURIComponent(sessionId)}/files/`;
    els.downloadReport.href = `${base}report.json`;
    els.downloadCsv.href = `${base}speaker_turns.csv`;
  }

  async function pollStatus() {
    try {
      currentStatus = await requestJson('/api/status');
      renderSessions(currentStatus.sessions || []);
      refreshButtons();
      if (currentStatus.recording) {
        els.time.textContent = fmtDuration(currentStatus.elapsed_seconds);
        if (Date.now() >= localErrorUntil) setStatus(`Recording ${currentStatus.session_id || ''}… Press Stop recording when you're done.`);
      } else if (Date.now() >= localErrorUntil && currentStatus.analyzing) {
        setStatus(currentStatus.message || 'Analysis in progress…');
      } else if (Date.now() >= localErrorUntil && currentStatus.error) {
        setStatus(currentStatus.error, true);
      } else if (Date.now() >= localErrorUntil && currentStatus.phase === 'complete') {
        setStatus(currentStatus.message || 'Analysis complete.');
      } else if (Date.now() >= localErrorUntil && currentStatus.phase === 'ready') {
        setStatus(currentStatus.message || 'Recording saved. Click Analyze when ready.');
      }
      if (currentStatus.session_id && currentStatus.phase === 'complete' && selectedSession === currentStatus.session_id) {
        const loadKey = `${currentStatus.session_id}:complete`;
        if (lastAutoLoaded !== loadKey) { lastAutoLoaded = loadKey; await loadSession(selectedSession); }
      }
      if (currentStatus.session_id && currentStatus.phase === 'ready' && selectedSession === currentStatus.session_id) {
        const loadKey = `${currentStatus.session_id}:ready`;
        if (lastAutoLoaded !== loadKey) { lastAutoLoaded = loadKey; await loadSession(selectedSession); }
      }
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  els.start.addEventListener('click', async () => {
    try {
      setStatus('Starting microphone…');
      await postJson('/api/record/start', { settings: settings() });
      lastAutoLoaded = '';
      await pollStatus();
    } catch (error) { setStatus(error.message, true); }
  });

  els.stop.addEventListener('click', async () => {
    try {
      els.stop.disabled = true;
      setStatus('Stopping microphone and saving recording…');
      const data = await postJson('/api/record/stop', {});
      selectedSession = data.session_id;
      sessionSignature = '';
      lastAutoLoaded = '';
      await pollStatus();
      await selectSession(data.session_id);
      setStatus('Recording saved. Click Analyze when ready.');
    } catch (error) { setStatus(error.message, true); await pollStatus(); }
  });

  els.analyze.addEventListener('click', async () => {
    if (!selectedSession) return;
    try {
      setStatus('Starting analysis…');
      const payload = { hf_token: tokenValue(), settings: settings() };
      await postJson(`/api/sessions/${encodeURIComponent(selectedSession)}/analyze`, payload);
      lastAutoLoaded = '';
      await pollStatus();
      setStatus('Analysis started. Results will appear here when it finishes.');
    } catch (error) { setStatus(error.message, true); }
  });

  els.device.addEventListener('change', () => {
    const compute = els.device.value === 'cuda' ? 'float16' : 'int8';
    if (els.device.value === 'cuda') setStatus('CUDA selected. This requires a working CUDA-enabled PyTorch installation.');
    else if (currentStatus?.phase === 'idle') setStatus(`CPU selected. Analysis will use INT8 precision.`);
  });
  els.refresh.addEventListener('click', async () => { sessionSignature = ''; await pollStatus(); });
  els.copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(els.transcript.textContent || ''); setStatus('Transcript copied.'); }
    catch { setStatus('Copy failed. Select the transcript text and copy it manually.', true); }
  });
  els.close.addEventListener('click', async () => {
    if (getBusy()) { setStatus('Stop the recording and wait for analysis to finish before closing the app.', true); return; }
    try {
      await postJson('/api/shutdown', {});
      document.body.innerHTML = '<main style="font:14px Segoe UI,Arial,sans-serif;max-width:640px;margin:80px auto;padding:24px"><h2>Application closed</h2><p>You can close this browser tab.</p></main>';
    } catch (error) { setStatus(error.message, true); }
  });

  pollStatus();
  window.setInterval(pollStatus, 900);
})();
