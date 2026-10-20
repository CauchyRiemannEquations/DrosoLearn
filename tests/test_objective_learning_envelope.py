"""Invented receipts only: actual cohort work starts after frozen publication."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drosolearn import objective_envelope as env

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'prospective_observed_objective_learning', 'external_manifest': None,
        'measurement_storage': 'lossless_deterministic_gzip_csv_crlf',
        'model_seeds': list(range(17801, 17811)), 'training_trials': 400,
        'checkpoints': [0, 50, 100, 200, 400], 'methods': list(env.METHODS),
        **{name: sorted(value) for name, value in (('source_files', env.SOURCE_FILES),
            ('parent_files', env.PARENT_FILES), ('graph_files', env.GRAPH_FILES), ('input_files', env.INPUT_FILES))},
        'interpretation': {key: False for key in ('new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim')}}


def summary():
    counts = {'training_feedback_trials': 192000, 'static_probe_cases': 9600,
        'model_initializations': 480, 'fixed_feature_constructions': 60, 'model_seeds': 10,
        'head_update_calls': 192000, 'head_nonzero_steps': 90000,
        'internal_update_calls': 0, 'value_update_calls': 0, 'internal_nonzero_steps': 0, 'value_nonzero_steps': 0,
        'auxiliary_memory_observation_calls': 96000, 'auxiliary_memory_discoveries': 1920,
        'preflight_direction_evaluations': 19440, 'runtime_selective_direction_evaluations': 24000,
        'balanced_direction_requests': 28000, 'always_balanced_direction_requests': 12000,
        'gated_balanced_direction_requests': 8000, 'gated_isolated_direction_evaluations': 8000,
        'gated_balanced_mode_trials': 16000, 'gated_gate_activations': 60,
        'self_blend_gate_activations': 60, 'self_blend_direction_requests': 8000,
        'self_blend_mode_trials': 16000, 'self_blend_isolated_direction_evaluations': 8000,
        'balanced_unique_solver_cases': 300, 'balanced_NoGo_solver_skips': 28000,
        'uniform_direction_requests': 8000, 'probability_direction_requests': 8000,
        'uniform_unique_solver_cases': 240, 'probability_unique_solver_cases': 8000,
        'objective_unique_solver_cases': 8240, 'total_unique_solver_cases': 8540,
        'uniform_gate_activations': 60, 'probability_gate_activations': 60,
        'uniform_mode_trials': 16000, 'probability_mode_trials': 16000,
        'uniform_isolated_direction_evaluations': 8000, 'probability_isolated_direction_evaluations': 8000,
        'objective_NoGo_solver_skips': 16000, 'unknown_other_destination_exposures': 2000,
        'source_inclusive_unknown_exposures_before_feedback': 5000,
        'source_inclusive_unknown_exposures_after_feedback': 3080,
        'lookup_static_cases': 9600, 'lookup_autonomous_training_trials': 0,
        'numeric_arrays': 215, 'measurement_columns': 96}
    return {'study_id': env.STUDY_ID, 'status': 'completed', 'counts': counts}


def test_exact_frozen_source_parent_graph_input_and_new_cohort():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert [len(x) for x in (env.SOURCE_FILES, env.PARENT_FILES, env.GRAPH_FILES, env.INPUT_FILES)] == [49, 14, 4, 2]
    assert 'src/drosolearn/self_blend_learning.py' in env.REUSED_SOURCE
    assert 'src/drosolearn/probability_envelope.py' in env.REUSED_SOURCE
    assert 'configs/self_blend_learning_v1.json' not in env.SOURCE_FILES
    assert 'results/probability_alignment_presentation_validation.json' in env.PARENT_FILES
    assert 'results/probability_alignment_archive_finalization_attempt01.zip' in env.PARENT_FILES
    assert 'measurements.csv.gz' in env.PRIMARY and 'measurements.csv' in env.PARENT_PRIMARY
    assert 'results/probability_alignment_v1/measurements.csv' in env.PARENT_FILES


@pytest.mark.parametrize('group', ['source_files', 'parent_files', 'graph_files', 'input_files'])
@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'extra'])
def test_exact_unique_closures_cannot_silently_change(group, mode):
    cfg = configuration()
    if mode == 'missing': cfg[group] = cfg[group][1:]
    elif mode == 'duplicate': cfg[group][-1] = cfg[group][0]
    else: cfg[group].append('unfrozen')
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('scope', ['integration_test', 'retrospective_descriptive_probability_alignment', None])
def test_saved_analysis_and_fixture_are_not_the_new_actual_study(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('key', ['new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim'])
def test_bounded_artificial_learning_interpretation(key):
    cfg = configuration(); cfg['interpretation'][key] = True
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_deterministic_lossless_gzip_preserves_original_CSV_UTF8_CRLF_bytes(tmp_path):
    rows = [{'text': '한글,쉼표', 'value': 1.25, 'nullable': None},
        {'text': 'embedded\nnewline', 'value': -2., 'nullable': 0}]
    plain = tmp_path / 'original.csv'; first = tmp_path / 'first.csv.gz'; second = tmp_path / 'second.csv.gz'
    env.csv_write(plain, rows)
    env.compressed_csv_write(first, rows); env.compressed_csv_write(second, rows)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[3] == 0 and first.read_bytes()[4:8] == b'\0' * 4
    assert gzip.decompress(first.read_bytes()) == plain.read_bytes()
    assert b'\r\n' in plain.read_bytes()
    with pytest.raises(FileExistsError): env.compressed_csv_write(first, rows)


def test_plain_CSV_storage_cannot_replace_fixed_new_gzip_storage():
    cfg = configuration(); cfg['measurement_storage'] = 'plain_csv'
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_exact_192000_actual_trials_9600_probes_and_dynamic_solver_accounting():
    env.validate_counts(configuration(), summary(), 192000)


@pytest.mark.parametrize('field,value', [
    ('training_feedback_trials', 144000), ('static_probe_cases', 7200), ('model_initializations', 360),
    ('fixed_feature_constructions', 61), ('head_update_calls', 0), ('head_nonzero_steps', 0),
    ('internal_update_calls', 1), ('value_update_calls', 1), ('internal_nonzero_steps', False),
    ('numeric_arrays', 201), ('measurement_columns', 87), ('lookup_autonomous_training_trials', 9600),
    ('balanced_direction_requests', 27999), ('balanced_unique_solver_cases', 28001),
    ('balanced_NoGo_solver_skips', 0), ('gated_isolated_direction_evaluations', 7999),
    ('self_blend_gate_activations', 61), ('uniform_direction_requests', 16001),
    ('uniform_unique_solver_cases', 241), ('probability_unique_solver_cases', 240),
    ('objective_unique_solver_cases', 0), ('total_unique_solver_cases', 8240),
    ('objective_NoGo_solver_skips', 0), ('uniform_mode_trials', True),
    ('probability_isolated_direction_evaluations', -1), ('auxiliary_memory_discoveries', 1921),
    ('source_inclusive_unknown_exposures_after_feedback', 3081), ('unknown_other_destination_exposures', 3081),
])
def test_work_axes_caches_probability_per_event_and_memory_counts_fail_closed(field, value):
    changed = summary(); changed['counts'][field] = value
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 192000)


@pytest.mark.parametrize('field,value', [('model_seeds', list(range(17701, 17711))),
    ('training_trials', 800), ('checkpoints', [0, 400]), ('methods', list(env.METHODS[:-1]))])
def test_old_cohort_larger_budget_or_changed_policy_count_cannot_replace_fixed_design(field, value):
    cfg = configuration(); cfg[field] = value
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 192000)


def test_no_full_observation_opportunities_are_preserved_without_forcing_positive_U_P_cases():
    result = summary(); c = result['counts']
    for prefix in ('uniform', 'probability'):
        for suffix in ('direction_requests', 'unique_solver_cases', 'gate_activations', 'mode_trials'):
            c[prefix + '_' + suffix] = 0
        c[prefix + '_isolated_direction_evaluations'] = 24000
    c['objective_unique_solver_cases'] = c['objective_NoGo_solver_skips'] = 0
    c['total_unique_solver_cases'] = c['balanced_unique_solver_cases']
    env.validate_counts(configuration(), result, 192000)


def test_unknown_counter_and_bool_row_count_do_not_pass():
    result = summary(); result['counts']['unregistered_solver_calls'] = 1
    with pytest.raises(ValueError): env.validate_counts(configuration(), result, 192000)
    with pytest.raises(ValueError): env.validate_counts(configuration(), summary(), True)


def test_parent_byte_gate_precedes_any_scientific_import(tmp_path, monkeypatch):
    from drosolearn import probability_envelope as prior
    def stop(root):
        assert root == tmp_path
        raise ValueError('original probability byte gate')
    monkeypatch.setattr(prior, 'parent_gate', stop)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature new science import'))
    with pytest.raises(ValueError, match='original probability byte gate'): env.parent_gate(tmp_path)


@pytest.mark.parametrize('gate', ['validate_registration', 'publication'])
def test_no_new_graph_seed_learner_or_solver_import_before_published_bytes(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    def stop(*args): raise ValueError('frozen private evidence missing')
    monkeypatch.setattr(env, gate, stop)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature new cohort work'))
    with pytest.raises(ValueError, match='frozen private evidence missing'):
        env.run(tmp_path, tmp_path / 'cfg', tmp_path / 'reg', tmp_path / 'proof', tmp_path / env.RESULT)
    assert not (tmp_path / env.RESULT).exists()


def run_fixture(tmp_path, monkeypatch, fail=False, bad_columns=False):
    (tmp_path / 'results').mkdir()
    for name in ('cfg.json', 'reg.json', 'proof.json'): (tmp_path / name).write_text('{}')
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {'source_git_commit': 'a' * 40}))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-01T00:00:00+00:00'}))
    def produce(*args):
        if fail: raise RuntimeError('invented objective solver failure')
        row = {str(i): 0 for i in range(95 if bad_columns else 96)}
        return {str(i): np.zeros(1) for i in range(215)}, [row] * 192000
    module = SimpleNamespace(validate_config=lambda *args: None, produce=produce, summarize=lambda *args: summary())
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: module)
    def fake_write(path, rows):
        assert len(rows) == 192000 and len(rows[0]) == 96
        path.write_bytes(b'only a fake serialization receipt\n')
    monkeypatch.setattr(env, 'compressed_csv_write', fake_write)
    return (tmp_path, tmp_path / 'cfg.json', tmp_path / 'reg.json', tmp_path / 'proof.json', tmp_path / env.RESULT)


def test_actual_learning_metadata_does_not_claim_saved_analysis_or_zero_solver_work(tmp_path, monkeypatch):
    args = run_fixture(tmp_path, monkeypatch)
    meta = env.run(*args)
    assert meta['status'] == 'completed' and meta['analysis_only'] is False
    assert meta['measurement_storage'] == 'lossless_deterministic_gzip_csv_crlf'
    assert meta['counts']['training_feedback_trials'] == 192000
    assert meta['counts']['total_unique_solver_cases'] == 8540
    assert meta['counts']['internal_update_calls'] == meta['counts']['value_update_calls'] == 0


@pytest.mark.parametrize('failure', ['solver', 'columns'])
def test_original_failed_run_is_exclusive_and_no_zero_completed_counts_are_invented(tmp_path, monkeypatch, failure):
    args = run_fixture(tmp_path, monkeypatch, fail=failure == 'solver', bad_columns=failure == 'columns')
    with pytest.raises((ValueError, RuntimeError)): env.run(*args)
    meta = json.loads((args[-1] / 'metadata.json').read_text())
    assert meta['status'] == 'failed' and 'counts' not in meta
    assert (args[-1] / 'registration.json').read_bytes() == args[2].read_bytes()
    assert 'metadata.json' in json.loads((args[-1] / 'artifact_hashes.json').read_text())
    with pytest.raises(ValueError): env.run(*args)
