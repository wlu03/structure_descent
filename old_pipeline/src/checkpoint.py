"""
Checkpoint system for Structure Descent pipeline.

Saves intermediate results after each completed stage so the pipeline
can resume from the last successful checkpoint on restart.

Checkpoint stages (in order):
  1. data_prep     — purchases_processed.parquet + train/val/test_choices.pkl
  2. features      — train_features.pkl
  3. initial_fit   — current_state.pkl (weights + score for initial structure)
  4. outer_loop    — checkpoint saved after EACH completed iteration
  5. final         — final_structure.pkl

Usage:
    ckpt = Checkpoint('data')
    if ckpt.has('data_prep'):
        data = ckpt.load('data_prep')
    else:
        data = run_data_prep(...)
        ckpt.save('data_prep', data)
"""

import pickle
import json
from pathlib import Path
from typing import Any, Optional


STAGES = ['data_prep', 'features', 'initial_fit', 'outer_loop', 'final']


class Checkpoint:
    """Manages pipeline checkpoints in a data directory."""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._manifest_path = self.data_dir / 'checkpoint_manifest.json'
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(self._manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)

    def has(self, stage: str) -> bool:
        """Check if a stage checkpoint exists and its files are present."""
        if stage not in self._manifest:
            return False
        files = self._manifest[stage].get('files', [])
        return all((self.data_dir / f).exists() for f in files)

    def save(self, stage: str, data: dict, files: list[str]):
        """
        Save checkpoint data and record which files belong to this stage.

        Args:
            stage: stage name (e.g. 'data_prep', 'outer_loop')
            data:  dict of objects to pickle (key -> object)
            files: list of filenames that were saved for this stage
        """
        for key, obj in data.items():
            path = self.data_dir / key
            if key.endswith('.parquet'):
                obj.to_parquet(path, index=False)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(obj, f)

        self._manifest[stage] = {'files': files, 'status': 'complete'}
        self._save_manifest()
        print(f'  [Checkpoint] Saved: {stage} ({len(files)} files)')

    def load_file(self, filename: str) -> Any:
        """Load a single file from the checkpoint directory."""
        import pandas as pd
        path = self.data_dir / filename
        if filename.endswith('.parquet'):
            return pd.read_parquet(path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_outer_loop_state(self) -> Optional[dict]:
        """
        Get the last completed outer loop iteration state.

        Returns dict with keys:
            iteration, structure, score, history
        Or None if no outer loop checkpoint exists.
        """
        path = self.data_dir / 'outer_loop_checkpoint.pkl'
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_outer_loop_iteration(self, iteration: int, structure, score: float,
                                  history: list, weights=None,
                                  proposal_log: Optional[list] = None):
        """
        Save checkpoint after a completed outer loop iteration.
        Only called after an iteration fully completes (fit + accept/reject).
        """
        state = {
            'iteration': iteration,
            'structure': structure.to_dict() if hasattr(structure, 'to_dict') else structure,
            'score': score,
            'history': history,
        }
        if weights is not None:
            state['weights'] = weights
        if proposal_log is not None:
            state['proposal_log'] = proposal_log

        with open(self.data_dir / 'outer_loop_checkpoint.pkl', 'wb') as f:
            pickle.dump(state, f)

        self._manifest['outer_loop'] = {
            'files': ['outer_loop_checkpoint.pkl'],
            'status': 'complete',
            'last_iteration': iteration,
        }
        self._save_manifest()
        print(f'  [Checkpoint] Outer loop iter {iteration} saved')

    def clear(self, stage: Optional[str] = None):
        """Clear a specific stage or all checkpoints."""
        if stage:
            files = self._manifest.get(stage, {}).get('files', [])
            for f in files:
                p = self.data_dir / f
                if p.exists():
                    p.unlink()
            self._manifest.pop(stage, None)
        else:
            for s in list(self._manifest.keys()):
                self.clear(s)
        self._save_manifest()

    def status(self) -> dict:
        """Return status of all pipeline stages."""
        result = {}
        for stage in STAGES:
            if self.has(stage):
                info = self._manifest[stage]
                if stage == 'outer_loop':
                    result[stage] = f"complete (iter {info.get('last_iteration', '?')})"
                else:
                    result[stage] = 'complete'
            else:
                result[stage] = 'not started'
        return result
