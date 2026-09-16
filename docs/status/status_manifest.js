'use strict';
function statusLines(manifest) {
  if (manifest.schema_version !== 2 || manifest.real_flight_validated !== false ||
      Object.keys(manifest.track_d).join(',') !== 'D1,D2,D3,D4,D5,D6,D7,D8,D9') {
    throw new Error('Unsupported status manifest');
  }
  return Object.entries(manifest.track_d).map(([id, row]) => `${id} · ${row.label}: ${row.status}`);
}
const LIFECYCLE = new Set(['COMPLETED', 'PLANNED', 'BLOCKED', 'NOT_TESTED', 'ARCHIVED_WITHDRAWN']);
function registryLines(registry) {
  if (registry.schema_version !== 1 ||
      Object.keys(registry.status_semantics).join(',') !==
        'COMPLETED,PLANNED,BLOCKED,NOT_TESTED,ARCHIVED_WITHDRAWN') {
    throw new Error('Unsupported research status registry');
  }
  return Object.entries(registry.components).map(([id, row]) => {
    if (!LIFECYCLE.has(row.lifecycle_status)) throw new Error('Unknown lifecycle status');
    return `${id} · ${row.label}: ${row.lifecycle_status} / ${row.evidence_status}`;
  });
}
function renderStatus(manifest, registry, node) {
  statusLines(manifest);
  registryLines(registry);
  const cells = [...document.querySelectorAll('[data-status-id]')];
  for (const cell of cells) {
    const row = manifest.track_d[cell.dataset.statusId];
    if (!row) throw new Error('Unknown status row');
    cell.textContent = row.status;
  }
  const components = [...document.querySelectorAll('[data-component-id]')];
  for (const cell of components) {
    const row = registry.components[cell.dataset.componentId];
    if (!row) throw new Error('Unknown component status row');
    cell.textContent = row.lifecycle_status;
    cell.dataset.lifecycle = row.lifecycle_status;
    const verdict = cell.parentElement && cell.parentElement.querySelector('[data-evidence-status]');
    if (verdict) verdict.textContent = row.evidence_status;
  }
  const link = document.createElement('a');
  link.href = '../status_manifest.json';
  link.textContent = 'Track D manifest';
  const registryLink = document.createElement('a');
  registryLink.href = '../research_status_registry.json';
  registryLink.textContent = 'component registry';
  node.replaceChildren(document.createTextNode('Status source: '), link,
    document.createTextNode(' + '), registryLink,
    document.createTextNode(' · lifecycle is separate from evidence verdict.'));
}
function failClosed(node) {
  for (const cell of document.querySelectorAll('[data-status-id]')) cell.textContent = 'UNAVAILABLE';
  for (const cell of document.querySelectorAll('[data-component-id]')) {
    cell.textContent = 'UNAVAILABLE';
    cell.dataset.lifecycle = 'UNAVAILABLE';
  }
  node.textContent = 'Status unavailable — consult VERIFICATION.md; no PASS inferred.';
}
if (typeof module !== 'undefined') module.exports = { statusLines, registryLines, renderStatus };
if (typeof document !== 'undefined') {
  Promise.all([
    fetch('../status_manifest.json'),
    fetch('../research_status_registry.json'),
  ]).then(responses => {
    if (responses.some(response => !response.ok)) throw new Error('Status source unavailable');
    return Promise.all(responses.map(response => response.json()));
  }).then(([manifest, registry]) =>
    renderStatus(manifest, registry, document.getElementById('public-status-manifest')))
    .catch(() => failClosed(document.getElementById('public-status-manifest')));
}
