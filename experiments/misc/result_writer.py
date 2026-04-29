import csv
import os
from pathlib import Path
from typing import Iterable, Sequence


class IncrementalCsvWriter:
    def __init__(
        self,
        path: str | Path,
        header: Sequence[str],
        *,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self.header = list(header)
        self.append = append
        self._handle = None
        self._writer = None

    def __enter__(self) -> "IncrementalCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        if self.append and file_exists and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as existing_handle:
                existing_header = next(csv.reader(existing_handle), [])
            if existing_header != self.header:
                raise ValueError(
                    f"CSV header mismatch for {self.path}: "
                    f"expected {self.header}, found {existing_header}."
                )

        mode = "a" if self.append else "w"
        self._handle = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if not self.append or not file_exists or self.path.stat().st_size == 0:
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
