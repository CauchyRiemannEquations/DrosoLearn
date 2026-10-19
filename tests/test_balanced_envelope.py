"""Targeted gates for a new actual learner, distinct from its saved-path parent."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import balanced_envelope as env


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
            'scope': 'prospective_observed_balanced_learning', 'external_manifest': None,
            'model_seeds': list(range(17501, 17511)), 'training_trials': 400,
            'checkpoints': [0, 50, 100, 200, 400],
            **{group: sorted(values) for group, values in (
                ('source_files', env.SOURCE_FILES), ('parent_files', env.PARENT_FILES),
                ('graph_files', env.GRAPH_FILES), ('input_files', env.INPUT_FILES))},
            'interpretation': {'new_human_experiment': False, 'shared_neural_mechanism_claim': False,
                               'education_effect_claim': False}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed', 'counts': {
        'training_feedback_trials': 96000, 'static_probe_cases': 4800,
        'model_initializations': 240, 'fixed_feature_constructions': 60,
        'model_seeds': 10, 'head_update_calls': 96000, 'head_nonzero_steps': 47000,
        'internal_update_calls': 0, 'value_update_calls': 0,
        'internal_nonzero_steps': 0, 'value_nonzero_steps': 0}}


def test_exact_actual_science_and_historical_closure():
    cfg = configuration()
    env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 35 and len(env.PARENT_FILES) == 11
    assert len(env.GRAPH_FILES) == 4 and len(env.INPUT_FILES) == 2
    assert {'scripts/verify_selective_spillover.py', 'scripts/verify_cone_frontier.py',
            'src/drosolearn/cone_geometry.py'} <= env.SOURCE_FILES
    assert {'results/cone_frontier_v1/arrays.npz',
            'results/cone_frontier_v1/measurements.csv'} <= env.PARENT_FILES


@pytest.mark.parametrize('group', ['source_files', 'parent_files', 'graph_files', 'input_files'])
@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'extra'])
def test_inventory_cannot_omit_duplicate_or_expand(group, mutation):
    cfg = configuration()
    if mutation == 'omit':
        cfg[group] = cfg[group][1:]
    elif mutation == 'duplicate':
        cfg[group][-1] = cfg[group][0]
    else:
        cfg[group].append('undeclared.data')
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('scope', ['integration_test', 'retrospective_descriptive_cone_frontier', None])
def test_fixture_or_prior_readonly_scope_cannot_be_registered(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('field', ['new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim'])
def test_interpretation_is_bounded(field):
    cfg = configuration(); cfg['interpretation'][field] = True
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_real_training_count_contract():
    env.validate_counts(configuration(), summary(), 96000)


@pytest.mark.parametrize('field,value', [
    ('training_feedback_trials', 0), ('static_probe_cases', 4000), ('model_initializations', 60),
    ('fixed_feature_constructions', 240), ('head_update_calls', 0), ('model_seeds', True),
    ('internal_update_calls', 1), ('internal_nonzero_steps', False), ('value_update_calls', 1),
    ('value_nonzero_steps', 1), ('head_nonzero_steps', 0), ('head_nonzero_steps', 96001),
    ('head_nonzero_steps', True),
])
def test_actual_work_cannot_be_relabelled_zero_learning_or_boolean(field, value):
    value_summary = summary(); value_summary['counts'][field] = value
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), value_summary, 96000)


def test_no_budget_extension_or_incomplete_rows():
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError):
        env.validate_counts(cfg, summary(), 96000)
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), summary(), 95999)


def test_actual_parent_gate_does_not_import_scientific_producers_or_draw_rng(monkeypatch):
    import numpy as np
    monkeypatch.setattr(np.random, 'default_rng', lambda *args, **kwargs: pytest.fail('parent RNG draw'))
    original = env.importlib.import_module
    def bounded_import(name, *args, **kwargs):
        assert name not in ('drosolearn.balanced_learning', 'drosolearn.cone_frontier',
                            'drosolearn.normalization_loss', 'drosolearn.selective_spillover')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(env.importlib, 'import_module', bounded_import)
    gate = env.parent_gate(ROOT)
    assert gate['baseline_tag'] == 'cone-frontier-v1'
    assert gate['prior_frontier_is_not_actual_learning'] is True
    assert gate['parent_gate_no_new_learner_graph_or_seed_calls'] is True
    assert gate['prior_negative_learning_and_human_results_preserved'] is True


def test_no_science_import_before_registration_gate(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (_ for _ in ()).throw(ValueError('freeze missing')))
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature science import'))
    with pytest.raises(ValueError, match='freeze missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_no_science_import_before_remote_publication_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    monkeypatch.setattr(env, 'publication', lambda *args: (_ for _ in ()).throw(ValueError('remote bytes missing')))
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature science import'))
    out = tmp_path / env.RESULT
    with pytest.raises(ValueError, match='remote bytes missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_failed_actual_run_is_exclusive_and_keeps_failure_metadata(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT; out.parent.mkdir()
    registration = tmp_path / 'reg.json'; registration.write_text('{}')
    config = tmp_path / 'cfg.json'; config.write_text('{}')
    proof = tmp_path / 'proof.json'; proof.write_text('{}')
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {'source_git_commit': '0' * 40}))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-01T00:00:00+00:00'}))
    def fail(*args):
        raise RuntimeError('invented actual-learning failure')
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: SimpleNamespace(validate_config=lambda *args: None, produce=fail))
    with pytest.raises(RuntimeError, match='invented actual-learning failure'):
        env.run(tmp_path, config, registration, proof, out)
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['status'] == 'failed' and metadata['analysis_only'] is False
    assert metadata['stage'] == 'actual_learning'
    assert 'counts' not in metadata
    assert (out / 'registration.json').read_bytes() == registration.read_bytes()
    assert 'metadata.json' in json.loads((out / 'artifact_hashes.json').read_text())
    with pytest.raises(ValueError, match='New fixed result directory'):
        env.run(tmp_path, config, registration, proof, out)


def proof_fixture(tmp_path, monkeypatch):
    registration = tmp_path / env.REGISTRATION
    registration.parent.mkdir(); registration.write_bytes(b'{"frozen":true}\n')
    proof = tmp_path / 'proof.json'
    reg = {'source_git_commit': 'a' * 40, 'registered_at_utc': '2020-01-01T00:00:00+00:00'}
    value = {'study_id': env.STUDY_ID, 'registration_sha256': env.sha(registration),
             'source_commit': reg['source_git_commit'], 'remote_bytes_match': True,
             'visibility': 'private', 'published_commit': 'b' * 40,
             'url': env.REPOSITORY + '/blob/' + 'b' * 40 + '/' + env.REGISTRATION,
             'checked_at_utc': '2020-01-02T00:00:00+00:00'}
    proof.write_text(json.dumps(value))
    monkeypatch.setattr(env, 'ancestor', lambda *args: None)
    monkeypatch.setattr(env, 'git', lambda *args: registration.read_bytes())
    return registration, reg, proof, value


def test_publication_binds_exact_private_bytes_commit_url_and_prior_time(tmp_path, monkeypatch):
    registration, reg, proof, value = proof_fixture(tmp_path, monkeypatch)
    assert env.publication(tmp_path, registration, reg, proof)[1] == value


@pytest.mark.parametrize('field,value', [
    ('remote_bytes_match', False), ('visibility', 'public'), ('registration_sha256', '0' * 64),
    ('source_commit', 'c' * 40), ('published_commit', 'not-a-commit'),
    ('url', 'https://example.test/not-registered-bytes'),
    ('checked_at_utc', '2019-12-31T00:00:00+00:00'),
])
def test_bad_publication_receipt_cannot_start_training(tmp_path, monkeypatch, field, value):
    registration, reg, proof, data = proof_fixture(tmp_path, monkeypatch)
    data[field] = value; proof.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        env.publication(tmp_path, registration, reg, proof)


def test_local_published_commit_bytes_must_equal_registration(tmp_path, monkeypatch):
    registration, reg, proof, _ = proof_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(env, 'git', lambda *args: b'changed')
    with pytest.raises(ValueError, match='registration bytes differ'):
        env.publication(tmp_path, registration, reg, proof)


def test_verify_rejects_readonly_metadata_before_raw_array_loading(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT; out.mkdir(parents=True)
    registration = tmp_path / 'reg.json'; registration.write_text('{}')
    config = tmp_path / 'cfg.json'; config.write_text('{}')
    proof = tmp_path / 'proof.json'; proof.write_text('{}')
    (out / 'registration.json').write_bytes(registration.read_bytes())
    reg = {'source_git_commit': 'a' * 40, 'registered_at_utc': '2020-01-01T00:00:00+00:00'}
    metadata = {'study_id': env.STUDY_ID, 'status': 'completed', 'stage': 'completed',
                'source_git_commit': reg['source_git_commit'], 'registration_sha256': env.sha(registration),
                'config_sha256': env.sha(config), 'publication_proof': 'proof.json',
                'publication_proof_sha256': env.sha(proof), 'analysis_only': True,
                'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
                'education_effect_tested': False}
    (out / 'metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), reg))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-02T00:00:00+00:00'}))
    with pytest.raises(ValueError, match='actual-learning identity'):
        env.verify_envelope(tmp_path, config, registration, out, proof)
