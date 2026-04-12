// ═══════════════════════════════════════════════════════════════
// MediMax Frontend Application Logic
// ═══════════════════════════════════════════════════════════════

const API_BASE = '/api';

// ── DOM Elements ───────────────────────────────────────────────
const UI = {
  statusDot: document.getElementById('status-dot'),
  statusText: document.getElementById('status-text'),
  
  navBtns: document.querySelectorAll('.nav-btn'),
  panels: document.querySelectorAll('.panel'),
  
  // Query Panel
  queryInput: document.getElementById('query-input'),
  btnQuery: document.getElementById('btn-query'),
  btnClear: document.getElementById('btn-clear'),
  exampleCards: document.querySelectorAll('.example-card'),
  loadingState: document.getElementById('loading-state'),
  answerCard: document.getElementById('answer-card'),
  confidenceBadge: document.getElementById('confidence-badge'),
  confidenceLabel: document.getElementById('confidence-label'),
  metaChips: document.getElementById('meta-chips'),
  answerBody: document.getElementById('answer-body'),
  sourcesSection: document.getElementById('sources-section'),
  sourcesList: document.getElementById('sources-list'),
  btnCopy: document.getElementById('btn-copy'),
  btnNewQuery: document.getElementById('btn-new-query'),
  
  // Filters
  filterDomain: document.getElementById('filter-domain'),
  filterYear: document.getElementById('filter-year'),
  filterK: document.getElementById('filter-k'),
  
  // Ingest Panel
  uploadZone: document.getElementById('upload-zone'),
  fileInput: document.getElementById('file-input'),
  btnBrowse: document.getElementById('btn-browse'),
  uploadProgress: document.getElementById('upload-progress'),
  progressBar: document.getElementById('progress-bar'),
  progressText: document.getElementById('progress-text'),
  uploadResults: document.getElementById('upload-results'),
  btnRefreshSources: document.getElementById('btn-refresh-sources'),
  sourcesPanelBody: document.getElementById('sources-panel-body'),
  
  // Stats Panel
  statChunks: document.getElementById('stat-chunks'),
  statDocs: document.getElementById('stat-docs'),
  statModel: document.getElementById('stat-model'),
  statLlm: document.getElementById('stat-llm'),
  cfgStrategy: document.getElementById('cfg-strategy'),
  cfgThreshold: document.getElementById('cfg-threshold'),
  cfgTopk: document.getElementById('cfg-topk'),
  cfgChunkSize: document.getElementById('cfg-chunk-size'),
  cfgChunkOverlap: document.getElementById('cfg-chunk-overlap'),
  syncIndicator: document.getElementById('sync-indicator'),
};

// ── Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkAPIHealth();
  setupNavigation();
  setupQueryEvents();
  setupUploadEvents();
  refreshStats();
  refreshSources();
  
  // Auto-resize textarea
  UI.queryInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });
});

// ── Toast Notifications ────────────────────────────────────────
const toastContainer = document.getElementById('toast-container');
function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✓' : '⚠️'}</span>
    <span>${message}</span>
  `;
  toastContainer.appendChild(toast);
  
  setTimeout(() => {
    toast.style.animation = 'toast-out 300ms ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── API Health Check ───────────────────────────────────────────
async function checkAPIHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      UI.statusDot.className = 'status-dot online';
      UI.statusText.textContent = 'System Online';
    } else {
      throw new Error('API Offline');
    }
  } catch (err) {
    UI.statusDot.className = 'status-dot error';
    UI.statusText.textContent = 'System Offline';
    showToast('Cannot connect to the MediMax backend API.', 'error');
  }
}

// ── Navigation ─────────────────────────────────────────────────
function setupNavigation() {
  UI.navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Update active nav
      UI.navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update active panel
      const targetId = `panel-${btn.dataset.panel}`;
      UI.panels.forEach(p => p.classList.remove('active'));
      document.getElementById(targetId).classList.add('active');
      
      if (btn.dataset.panel === 'stats') refreshStats();
      if (btn.dataset.panel === 'ingest') refreshSources();
    });
  });
}

// ── Query Logic ────────────────────────────────────────────────
function setupQueryEvents() {
  UI.btnQuery.addEventListener('click', submitQuery);
  UI.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });
  
  UI.btnClear.addEventListener('click', () => {
    UI.queryInput.value = '';
    UI.queryInput.style.height = 'auto';
    UI.queryInput.focus();
  });
  
  UI.exampleCards.forEach(card => {
    card.addEventListener('click', () => {
      UI.queryInput.value = card.dataset.query;
      submitQuery();
    });
  });

  UI.btnNewQuery.addEventListener('click', () => {
    UI.answerCard.classList.add('hidden');
    UI.queryInput.value = '';
    UI.queryInput.focus();
  });

  UI.btnCopy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(UI.answerBody.innerText);
      showToast('Answer copied to clipboard');
    } catch {
      showToast('Failed to copy', 'error');
    }
  });
}

async function submitQuery() {
  const query = UI.queryInput.value.trim();
  if (!query) return;

  // UI State: Loading
  UI.btnQuery.disabled = true;
  UI.answerCard.classList.add('hidden');
  UI.loadingState.classList.remove('hidden');
  
  const payload = {
    question: query,
    k: parseInt(UI.filterK.value) || 6,
    domain: UI.filterDomain.value || null,
    year_from: parseInt(UI.filterYear.value) || null
  };

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Query failed');
    }
    
    const data = await res.json();
    renderAnswer(data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    UI.loadingState.classList.add('hidden');
    UI.btnQuery.disabled = false;
  }
}

function processMarkdown(text) {
  // Simple bold and newline processing
  let processed = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  processed = processed.replace(/\n/g, '<br/>');
  return processed;
}

function renderAnswer(data) {
  // Render text
  UI.answerBody.innerHTML = processMarkdown(data.answer);
  
  // Render Confidence
  UI.confidenceBadge.className = 'confidence-badge';
  let confText = 'Unknown';
  
  if (data.confidence === 'High') {
    UI.confidenceBadge.classList.add('high');
    confText = 'High Confidence';
  } else if (data.confidence === 'Medium') {
    UI.confidenceBadge.classList.add('medium');
    confText = 'Medium Confidence';
  } else {
    UI.confidenceBadge.classList.add('low');
    confText = data.low_confidence_fallback ? 'No Sufficient Data' : 'Low Confidence';
  }
  
  const scorePct = data.max_similarity_score != null ? (data.max_similarity_score * 100).toFixed(1) : '0.0';
  UI.confidenceLabel.textContent = `${confText} (${scorePct}%)`;
  
  // Render Time Meta
  UI.metaChips.innerHTML = `<span class="chip">⏱ ${(data.processing_time || 0).toFixed(2)}s</span>`;
  
  // Render Sources
  UI.sourcesList.innerHTML = '';
  if (data.sources && data.sources.length > 0) {
    UI.sourcesSection.classList.remove('hidden');
    data.sources.forEach((src, idx) => {
      const el = document.createElement('div');
      el.className = 'source-item';
      
      const domainHtml = src.metadata.domain ? `<span class="source-domain">${src.metadata.domain}</span>` : '';
      const yearHtml = src.metadata.year ? ` • Year: ${src.metadata.year}` : '';
      
      el.innerHTML = `
        <div class="source-badge">${idx + 1}</div>
        <div class="source-info">
          <div class="source-name">${src.metadata.source || 'Unknown Source'} ${domainHtml}</div>
          <div class="source-meta">
            Relevance: ${(src.score * 100).toFixed(1)}%
            ${src.metadata.page ? ' • Page ' + src.metadata.page : ''}
            ${yearHtml}
          </div>
        </div>
      `;
      UI.sourcesList.appendChild(el);
    });
  } else {
    UI.sourcesSection.classList.add('hidden');
  }

  UI.answerCard.classList.remove('hidden');
}

// ── Upload Logic ───────────────────────────────────────────────
function setupUploadEvents() {
  UI.btnBrowse.addEventListener('click', () => UI.fileInput.click());
  UI.fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
  
  // Drag and drop
  UI.uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    UI.uploadZone.classList.add('drag-over');
  });
  UI.uploadZone.addEventListener('dragleave', e => {
    e.preventDefault();
    UI.uploadZone.classList.remove('drag-over');
  });
  UI.uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    UI.uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  });
  
  UI.btnRefreshSources.addEventListener('click', refreshSources);
}

async function handleFiles(files) {
  if (files.length === 0) return;
  
  UI.uploadResults.innerHTML = '';
  UI.uploadProgress.classList.remove('hidden');
  
  let successCount = 0;
  let total = files.length;
  
  for (let i = 0; i < total; i++) {
    const file = files[i];
    UI.progressText.textContent = `Processing ${i + 1} of ${total}: ${file.name}`;
    UI.progressBar.style.width = `${((i) / total) * 100}%`;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_BASE}/ingest/file`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) throw new Error(`Upload failed (${res.status})`);
      
      const data = await res.json();
      addUploadResult(file.name, true, `Ingested ${data.chunks_created || 0} chunks`);
      successCount++;
    } catch (err) {
      addUploadResult(file.name, false, err.message);
    }
  }
  
  UI.progressBar.style.width = '100%';
  UI.progressText.textContent = `Complete. ${successCount} of ${total} files ingested successfully.`;
  
  setTimeout(() => {
    UI.uploadProgress.classList.add('hidden');
    UI.progressBar.style.width = '0%';
    refreshSources();
    refreshStats();
  }, 3000);
  
  // Reset input
  UI.fileInput.value = '';
}

function addUploadResult(filename, success, detail) {
  const el = document.createElement('div');
  el.className = `result-item ${success ? 'success' : 'error'}`;
  el.innerHTML = `
    <div class="result-icon">${success ? '✓' : '⚠️'}</div>
    <div class="result-info">
      <div class="result-name">${filename}</div>
      <div class="result-detail">${detail}</div>
    </div>
  `;
  UI.uploadResults.appendChild(el);
}

// ── Stats & Sources Data Fetching ──────────────────────────────
async function refreshStats() {
  try {
    // Stats is now fast — chunk count only, no source list
    const res = await fetch(`${API_BASE}/ingest/stats`);
    if (!res.ok) return;
    const data = await res.json();
    
    UI.statChunks.textContent = data.total_chunks.toLocaleString();
    UI.statModel.textContent = data.embedding_mode;
    UI.statLlm.textContent = "mistral-large";

    // Handle sync indicator based on chunk count
    handleSyncIndicator(data.total_chunks);

    // Fetch source count separately (can be slow on large DBs — non-blocking)
    fetchSourceCount();
    
  } catch (err) {
    console.error('Failed to fetch stats:', err);
  }
}

// Fetch just the count of unique sources — called from refreshStats non-blocking
async function fetchSourceCount() {
  try {
    const res = await fetch(`${API_BASE}/ingest/sources`);
    if (!res.ok) return;
    const data = await res.json();
    if (UI.statDocs) UI.statDocs.textContent = data.count ?? data.sources.length;
  } catch (err) {
    // Non-critical — sources list can be slow; just leave the old value
    console.warn('Could not fetch source count:', err);
  }
}


let lastChunkCount = -1;
let syncTimeout = null;

function handleSyncIndicator(currentCount) {
  if (lastChunkCount === -1) {
    lastChunkCount = currentCount;
    return;
  }

  if (currentCount > lastChunkCount) {
    // Count increased! Show syncing indicator
    UI.syncIndicator.classList.remove('hidden');
    
    // Clear existing timeout
    if (syncTimeout) clearTimeout(syncTimeout);
    
    // Hide after 30 seconds of no change
    syncTimeout = setTimeout(() => {
      UI.syncIndicator.classList.add('hidden');
    }, 30000);
    
    // Poll more frequently while syncing
    setTimeout(refreshStats, 5000);
  }
  
  lastChunkCount = currentCount;
}

async function refreshSources() {
  try {
    const res = await fetch(`${API_BASE}/ingest/sources`);
    if (!res.ok) return;
    const data = await res.json();
    
    UI.sourcesPanelBody.innerHTML = '';
    if (data.sources.length === 0) {
      UI.sourcesPanelBody.innerHTML = '<p class="empty-state">No documents indexed yet.</p>';
      return;
    }
    
    data.sources.forEach(src => {
      const el = document.createElement('div');
      el.className = 'source-row';
      el.innerHTML = `
        <div class="source-row-name">${src}</div>
        <button class="source-row-del" data-source="${src}">Delete</button>
      `;
      UI.sourcesPanelBody.appendChild(el);
    });
    
    // Attach delete handlers
    document.querySelectorAll('.source-row-del').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const sourceName = e.target.dataset.source;
        deleteSource(sourceName);
      });
    });
    
  } catch (err) {
    console.error('Failed to fetch sources:', err);
  }
}

async function deleteSource(sourceName) {
  if (!confirm(`Are you sure you want to delete all chunks for "${sourceName}"?`)) return;
  
  try {
    const res = await fetch(`${API_BASE}/ingest/source`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_name: sourceName })
    });
    
    if (!res.ok) throw new Error('Deletion failed');
    const data = await res.json();
    
    showToast(`Deleted ${data.chunks_deleted} chunks for ${sourceName}`);
    refreshSources();
    refreshStats();
  } catch (err) {
    showToast(err.message, 'error');
  }
}
