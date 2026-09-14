'use strict';
function statusLines(manifest) {
  if (manifest.schema_version !== 2 || manifest.real_flight_validated !== false ||
      Object.keys(manifest.track_d).join(',') !== 'D1,D2,D3,D4,D5,D6,D7,D8,D9') {
    throw new Error('Unsupported status manifest');
  }
  return Object.entries(manifest.track_d).map(([id, row]) => `${id} · ${row.label}: ${row.status}`);
}
function renderStatus(manifest, node) {
  statusLines(manifest);
  const cells = [...document.querySelectorAll('[data-status-id]')];
  for (const cell of cells) {
    const row = manifest.track_d[cell.dataset.statusId];
    if (!row) throw new Error('Unknown status row');
    cell.textContent = row.status;
  }
  const link = document.createElement('a');
  link.href = '../status_manifest.json';
  link.textContent = 'Track D manifest';
  node.replaceChildren(document.createTextNode('Status source: '), link,
    document.createTextNode(' · schema 2 · D1–D9 evidence links are preserved.'));
}
function failClosed(node) {
  for (const cell of document.querySelectorAll('[data-status-id]')) cell.textContent = 'UNAVAILABLE';
  node.textContent = 'Status unavailable — consult VERIFICATION.md; no PASS inferred.';
}
if (typeof module !== 'undefined') module.exports = { statusLines, renderStatus };
if (typeof document !== 'undefined') {
  fetch('../status_manifest.json').then(response => {
    if (!response.ok) throw new Error('Status manifest unavailable');
    return response.json();
  }).then(manifest => renderStatus(manifest, document.getElementById('public-status-manifest')))
    .catch(() => failClosed(document.getElementById('public-status-manifest')));
}
