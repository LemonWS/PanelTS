from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


@dataclass
class HubClient:
    repo_id: str = "Multiple-Time-Series/PanelTS"
    revision: str = "main"
    cache_dir: str | Path | None = None
    token: str | bool | None = None
    local_files_only: bool = False
    _files: list[str] | None = field(default=None, init=False, repr=False)

    def list_files(self, prefix: str | None = None, suffix: str | None = None) -> list[str]:
        if self._files is None:
            api = HfApi(token=self.token)
            self._files = list(
                api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    token=self.token,
                )
            )
        files = self._files
        if prefix is not None:
            p = prefix.rstrip("/") + "/"
            files = [f for f in files if f.startswith(p)]
        if suffix is not None:
            files = [f for f in files if f.lower().endswith(suffix.lower())]
        return sorted(files)

    def download(self, path_in_repo: str) -> str:
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=self.revision,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            token=self.token,
            local_files_only=self.local_files_only,
        )

    def read_csv(self, path_in_repo: str, **kwargs: Any) -> pd.DataFrame:
        path = self.download(path_in_repo)
        options = {"encoding": "utf-8-sig"}
        options.update(kwargs)
        try:
            return pd.read_csv(path, **options)
        except UnicodeDecodeError:
            options["encoding"] = "latin1"
            return pd.read_csv(path, **options)
