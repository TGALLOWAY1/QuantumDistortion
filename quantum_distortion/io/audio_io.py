from pathlib import Path

from typing import Tuple, Union, Optional

import numpy as np

import soundfile as sf



def load_audio(path: Union[str, Path], target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:

    path = Path(path)

    audio, sr = sf.read(path, always_2d=False)

    return audio.astype(np.float32), sr



def save_audio(path: Union[str, Path], audio: np.ndarray, sr: int) -> None:

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(path, audio, sr)
