"""CPU-only audit of archived P10 counts and the exploratory capture interaction."""
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    summary = json.loads(Path(__file__).with_name('summary.json').read_text())
    cells = summary['cells']
    assert len(cells) == 8
    for name, cell in cells.items():
        path = root / cell['receipt']
        result = json.loads(path.read_text())
        receipt = json.loads(path.with_name('205bars.receipt.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['result_sha256'], name
        assert result['actual_episodes'] == cell['episodes'], name
        assert sum(cell[k] for k in ('captured', 'crash', 'timeout')) == cell['episodes'], name
        for key in ('captured', 'crash', 'timeout'):
            assert result['outcome'][key] == cell[key], (name, key)
        for field, expected in (
            ('p9_empirical_error_enabled', cell['p9_enabled']),
            ('p9_empirical_error_sha256', cell['p9_model_sha256']),
            ('p9_empirical_error_seed', cell['p9_seed']),
        ):
            assert result['condition'][field] == expected, (name, field)
        assert receipt['runtime_git_dirty'] is False, name
        assert receipt['runtime_git_commit'] == cell['runtime_git_commit'], name
        assert receipt['source_checkpoint_sha256'] == cell['checkpoint_sha256'], name

    def pooled(arm, outcome):
        selected = [c for name, c in cells.items() if name.startswith(arm + '_s')]
        n = sum(c['episodes'] for c in selected)
        return sum(c[outcome] for c in selected) / n, n

    contrasts = {
        'primary': ('adapted_p9', 'source_p9'),
        'secondary_clean_retention': ('adapted_clean', 'source_clean'),
        'reference_cost_of_p9_error_on_source': ('source_p9', 'source_clean'),
        'reference_cost_of_p9_error_on_adapted': ('adapted_p9', 'adapted_clean'),
    }
    z = 1.959963984540054
    for key, (a, b) in contrasts.items():
        for outcome in ('captured', 'crash', 'timeout'):
            pa, na = pooled(a, outcome)
            pb, nb = pooled(b, outcome)
            delta = 100 * (pa - pb)
            se = 100 * math.sqrt(pa * (1-pa) / na + pb * (1-pb) / nb)
            recorded = summary[key][outcome]
            assert abs(recorded['delta_pp'] - delta) < .00051, (key, outcome)
            for actual, expected in zip(recorded['ci95'], (delta-z*se, delta+z*se)):
                assert abs(actual-expected) < .00051, (key, outcome)

    terms = [(sign, *pooled(arm, 'captured')) for arm, sign in (
        ('adapted_p9', 1), ('adapted_clean', -1), ('source_p9', -1), ('source_clean', 1))]
    delta = 100 * sum(sign*p for sign, p, n in terms)
    se = 100 * math.sqrt(sum(p*(1-p)/n for sign, p, n in terms))
    print(json.dumps({
        'audit': 'PASS', 'cells': len(cells),
        'exploratory_capture_interaction_pp': delta,
        'ci95_pp': [delta-z*se, delta+z*se],
        'method': 'independent-cell Bernoulli normal approximation; covariance not estimated',
        'preregistered_primary': False, 'multiplicity_adjusted': False,
    }, indent=2))


if __name__ == '__main__':
    main()
