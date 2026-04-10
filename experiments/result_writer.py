import csv
import os
from pathlib import Path
from typing import Iterable, Sequence


class IncrementalCsvWriter:
    def __init__(
        self,
        path: str | Path,
        header: Sequence[str],
    ) -> None:
        self.path = Path(path)
        self.header = list(header)
        self._handle = None
        self._writer = None

    def __enter__(self) -> "IncrementalCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(self.header)
        self._sync()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None

    def write_row(self, row: Sequence[object]) -> None:
        if self._writer is None:
            raise RuntimeError("CSV writer is not open.")
        self._writer.writerow(list(row))
        self._sync()

    def write_rows(self, rows: Iterable[Sequence[object]]) -> None:
        for row in rows:
            self.write_row(row)

    def _sync(self) -> None:
        if self._handle is None:
            raise RuntimeError("CSV writer handle is not open.")
        self._handle.flush()
        os.fsync(self._handle.fileno())
