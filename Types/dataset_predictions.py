import os
import pickle
import numpy as np


class DatasetPredictions:
    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        timestamps: np.ndarray,
        predictor_name: str,
        num_historic_tokens: float,
    ):
        assert timestamps.ndim == 2, "Timestamps must be a 2D numpy array."
        assert lats.ndim == 2, "Lats must be a 2D numpy array."
        assert lons.ndim == 2, "Lons must be a 2D numpy array."
        assert timestamps.shape[0] == lats.shape[0] == lons.shape[0], \
            "All arrays must have the same number of trajectories."

        self._lons = lons
        self._lats = lats
        self._timestamps = timestamps
        self._predictor_name = predictor_name
        self._num_historic_tokens = num_historic_tokens

    @property
    def lats(self) -> np.ndarray:
        return self._lats

    @property
    def lons(self) -> np.ndarray:
        return self._lons

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def predictor_name(self) -> str:
        return self._predictor_name

    @property
    def num_historic_tokens(self) -> float:
        return self._num_historic_tokens

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving predictions to '{path}.npz'")
        np.savez_compressed(
            path,
            lats=self._lats,
            lons=self._lons,
            timestamps=self._timestamps,
            predictor_name=pickle.dumps(self._predictor_name),
            num_historic_tokens=pickle.dumps(self._num_historic_tokens),
        )
        print(f"Saved predictions of {len(self._lats):,} trajectories\n")

    @staticmethod
    def load(path: str) -> "DatasetPredictions":
        print(f"Loading predictions from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            lats = data['lats']
            lons = data['lons']
            timestamps = data['timestamps']
            predictor_name = pickle.loads(data['predictor_name'].item())
            num_historic_tokens = pickle.loads(
                data['num_historic_tokens'].item())

        return DatasetPredictions(
            lats=lats,
            lons=lons,
            timestamps=timestamps,
            predictor_name=predictor_name,
            num_historic_tokens=num_historic_tokens,
        )
