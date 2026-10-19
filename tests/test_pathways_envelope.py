"""New saved-path scope/count gates; inherited publication gates stay frozen."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import pathways_envelope as env


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
            'scope': 'retrospective_descriptive_balanced_pathways', 'external_manifest': None,
            'model_seeds': list(range(17501, 17511)), 'training_trials': 400,
            **{group: sorted(values) for group, values in (
                ('source_files', env.SOURCE_FILES), ('parent_files', env.PARENT_FILES))},
            'interpretation': {key: False for key in (
                'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim',
                'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}


def summary():
    return {'study_id': env.STUDY_ID, 'scope': 'retrospective_descriptive_balanced_pathways',
            'status': 'completed', 'counts': {
                'analyzed_parent_trial_records': 96000, 'analyzed_destination_effects': 384000,
                'paired_trial_records': 24000, 'measurement_records': 1440, 'numeric_arrays': 19,
                'bridge_eligible_trials': 12000, 'bridge_evaluated_corner_destination_effects': 384000,
                'full_known_objective_trials': 10000,
                **{key: 0 for key in ('new_training_trials', 'new_head_update_calls',
                    'new_internal_update_calls', 'new_value_update_calls', 'new_random_draws',
                    'fresh_model_constructions', 'new_optimization_calls')}}}


def test_exact_new_science_and_historical_closure():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 17 and len(env.PARENT_FILES) == 11
    assert {'src/drosolearn/balanced_envelope.py', 'src/drosolearn/cone_envelope.py',
            'src/drosolearn/pathways_envelope.py'} <= env.SOURCE_FILES
    assert {'results/balanced_learning_v1/arrays.npz',
            'results/balanced_learning_v1/measurements.csv'} <= env.PARENT_FILES
    assert 'src/drosolearn/balanced_learning.py' not in env.REUSED_SOURCE


@pytest.mark.parametrize('group', ['source_files', 'parent_files'])
@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'extra'])
def test_exact_inventory_cannot_omit_duplicate_or_expand(group, mutation):
    cfg = configuration()
    if mutation == 'omit':
        cfg[group] = cfg[group][1:]
    elif mutation == 'duplicate':
        cfg[group][-1] = cfg[group][0]
    else:
        cfg[group].append('undeclared.data')
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('field', ['new_training', 'new_seed_draws', 'new_effect_hypothesis_test',
                                 'shared_neural_mechanism_claim'])
def test_readonly_scope_cannot_be_upgraded(field):
    cfg = configuration(); cfg['interpretation'][field] = True
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_exact_saved_path_counts():
    env.validate_counts(configuration(), summary(), 1440)


@pytest.mark.parametrize('field,value', [
    ('analyzed_parent_trial_records', 24000), ('analyzed_destination_effects', 96000),
    ('paired_trial_records', 96000), ('measurement_records', 1260), ('numeric_arrays', 190),
    ('new_training_trials', 1), ('new_head_update_calls', 1), ('new_internal_update_calls', False),
    ('new_value_update_calls', 1), ('new_random_draws', 1), ('fresh_model_constructions', 1),
    ('new_optimization_calls', 1), ('bridge_eligible_trials', -1), ('bridge_eligible_trials', 24001),
    ('bridge_eligible_trials', True), ('bridge_evaluated_corner_destination_effects', 12000),
    ('full_known_objective_trials', 12001), ('full_known_objective_trials', -1),
])
def test_work_counts_reject_scope_expansion_rng_models_solver_boolean_or_bad_axes(field, value):
    value_summary = summary(); value_summary['counts'][field] = value
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), value_summary, 1440)


def test_extra_count_or_incomplete_measurement_table_fails():
    value_summary = summary(); value_summary['counts']['unregistered'] = 1
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), value_summary, 1440)
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), summary(), 1439)


def test_saved_parent_cohort_or_budget_cannot_change():
    cfg = configuration(); cfg['model_seeds'][0] = 17401
    with pytest.raises(ValueError):
        env.validate_counts(cfg, summary(), 1440)
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError):
        env.validate_counts(cfg, summary(), 1440)


@pytest.mark.parametrize('scope', ['integration_test', 'prospective_observed_balanced_learning', None])
def test_new_learning_or_fixture_cannot_enter_registered_saved_path_audit(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_inherited_parent_gate_is_byte_only_and_runs_first(tmp_path, monkeypatch):
    from drosolearn import balanced_envelope as prior
    def stop_at_gate(root):
        assert root == tmp_path
        raise RuntimeError('inherited byte gate reached')
    monkeypatch.setattr(prior, 'parent_gate', stop_at_gate)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('scientific import'))
    with pytest.raises(RuntimeError, match='inherited byte gate reached'):
        env.parent_gate(tmp_path)


def test_no_arithmetic_import_before_registration_gate(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (_ for _ in ()).throw(ValueError('freeze missing')))
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature scientific import'))
    with pytest.raises(ValueError, match='freeze missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_no_arithmetic_import_before_publication_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    monkeypatch.setattr(env, 'publication', lambda *args: (_ for _ in ()).throw(ValueError('remote bytes missing')))
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature scientific import'))
    out = tmp_path / env.RESULT
    with pytest.raises(ValueError, match='remote bytes missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_failed_audit_is_exclusive_and_preserves_zero_learning_metadata(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT; out.parent.mkdir()
    registration = tmp_path / 'reg.json'; registration.write_text('{}')
    config = tmp_path / 'cfg.json'; config.write_text('{}')
    proof = tmp_path / 'proof.json'; proof.write_text('{}')
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {'source_git_commit': '0' * 40}))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-01T00:00:00+00:00'}))
    def fail(*args):
        raise RuntimeError('invented saved-path failure')
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: SimpleNamespace(validate_config=lambda *args: None, produce=fail))
    with pytest.raises(RuntimeError, match='invented saved-path failure'):
        env.run(tmp_path, config, registration, proof, out)
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['status'] == 'failed' and metadata['analysis_only'] is True
    assert metadata['new_training_calls'] == metadata['new_random_seed_draws'] == metadata['new_graph_or_feature_constructions'] == 0
    assert metadata['new_optimization_calls'] == 0
    assert metadata['new_effect_hypothesis_test'] is False
    assert (out / 'registration.json').read_bytes() == registration.read_bytes()
    assert 'metadata.json' in json.loads((out / 'artifact_hashes.json').read_text())
    with pytest.raises(ValueError, match='New fixed result directory'):
        env.run(tmp_path, config, registration, proof, out)


def test_new_result_path_cannot_be_old_parent_path(tmp_path):
    out = tmp_path / 'results/balanced_learning_v1'
    with pytest.raises(ValueError, match='New fixed result directory'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
