# Changelog

## [0.4.4] - 2026-08-29

- Request buffers with `PyBUF_RECORDS` / `PyBUF_RECORDS_RO` (do not mix `PyBUF_READ`/`PyBUF_WRITE` into `GetBuffer`).
- `data` works for Fortran-contiguous arrays; `isContiguous` remains C-order only.
- 1D `[]` requires `ndim == 1` and supports strided / reversed 1D views.
- `=wasMoved` clears the buffer pointer so moved-from wrappers do not double-release.
- Document that wrappers consume existing buffers and do not allocate ndarrays.

## [0.4.3] - 2026-02-14

Current Nimble / GitHub release.

- `{.nuwa_export.}` writes stub JSON under `-d:nuwaStubDir=` when nuwa-build provides that path, with `NUWA_STUB:` stdout as fallback.
- `withNogil` for releasing the Python GIL around pure Nim work.
- NumPy buffer wrappers (`asNumpyArray`) with `PyBUF_FORMAT` included in buffer requests.

## [0.4.2] - 2026-02-13

Previous tagged release.
