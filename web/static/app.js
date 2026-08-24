const form = document.querySelector('#analysis-form');
const dateInput = document.querySelector('#trade-date');
const submit = document.querySelector('#submit');
const statusCard = document.querySelector('#status-card');
const resultCard = document.querySelector('#result-card');
let timer;

dateInput.value = new Date().toISOString().slice(0, 10);
dateInput.max = dateInput.value;

function showStatus(status, text) {
  statusCard.classList.remove('hidden');
  document.querySelector('#status-label').textContent = status.toUpperCase();
  document.querySelector('#status-title').textContent = status === 'running' ? 'Agenten analysieren den Markt' : 'Analyse wird vorbereitet';
  document.querySelector('#status-detail').textContent = text;
}
function renderText(text) { const node = document.createElement('p'); node.textContent = text; return node; }
function renderResult(result) {
  resultCard.classList.remove('hidden');
  document.querySelector('#signal').textContent = result.signal || 'Kein eindeutiges Signal';
  const decision = document.querySelector('#decision'); decision.replaceChildren(renderText(result.final_decision || 'Keine Entscheidung verfügbar.'));
  const reports = document.querySelector('#reports'); reports.replaceChildren();
  Object.entries(result.reports || {}).forEach(([name, text]) => { const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = name.replace('_', ' '); details.append(summary, renderText(text)); reports.append(details); });
}
async function poll(id) {
  const response = await fetch(`/api/analyses/${id}`); const job = await response.json();
  if (!response.ok) throw new Error(job.detail || 'Analyse konnte nicht geladen werden.');
  if (job.status === 'completed') { clearInterval(timer); submit.disabled = false; statusCard.classList.add('hidden'); renderResult(job.result); return; }
  if (job.status === 'failed') { clearInterval(timer); submit.disabled = false; showStatus('Fehler', job.error || 'Die Analyse ist fehlgeschlagen.'); document.querySelector('#spinner').classList.add('hidden'); return; }
  showStatus(job.status, job.status === 'running' ? 'Markt-, Nachrichten- und Risiko-Agenten arbeiten. Diese Seite aktualisiert sich automatisch.' : 'Die Analyse wartet auf einen freien Ausführungsplatz.');
}
form.addEventListener('submit', async (event) => {
  event.preventDefault(); clearInterval(timer); resultCard.classList.add('hidden'); submit.disabled = true;
  const analysts = [...document.querySelectorAll('[name=analysts]:checked')].map((item) => item.value);
  const body = { ticker: form.ticker.value, trade_date: form.trade_date.value, asset_type: form.asset_type.value, analysts };
  try { const response = await fetch('/api/analyses', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Bitte überprüfe deine Eingaben.'); showStatus('Warteschlange', 'Die Analyse wartet auf einen freien Ausführungsplatz.'); document.querySelector('#spinner').classList.remove('hidden'); await poll(data.id); timer = setInterval(() => poll(data.id).catch((error) => showStatus('Fehler', error.message)), 2000); } catch (error) { submit.disabled = false; showStatus('Fehler', error.message); document.querySelector('#spinner').classList.add('hidden'); }
});
