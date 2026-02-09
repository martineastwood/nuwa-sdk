## =============================================================================
## Test Helper Module for nuwa_sdk
## =============================================================================
## This module exports test functions that exercise the numpy wrappers
## so they can be tested via Python's pytest framework.

import nimpy
include nuwa_sdk

# =============================================================================
# Test Functions for NumPy Wrappers
# =============================================================================

proc test_1d_array_sum*(arr: PyObject): int64 {.nuwa_export.} =
  ## Test helper: Sum a 1D numpy array of int64
  ## Used to verify basic wrapper functionality
  var npArr = asNumpyArray(arr, int64)

  result = 0
  for val in items(npArr):
    result += val

proc test_1d_array_with_gil*(arr: PyObject): int64 {.nuwa_export.} =
  ## Test helper: Sum array with GIL release
  ## Verifies that withNogil works correctly with numpy wrappers
  var npArr = asNumpyArray(arr, int64)
  let n = npArr.len
  let data = npArr.data

  withNogil:
    var sum = 0'i64
    for i in 0..<n:
      sum += data[i]
    return sum

proc test_1d_array_multiply_scalar*(arr: PyObject, scalar: float64): seq[float64] {.nuwa_export.} =
  ## Test helper: Multiply array by scalar
  var npArr = asNumpyArray(arr, float64)

  result = newSeq[float64](npArr.len)
  for i, val in pairs(npArr):
    result[i] = val * scalar

proc test_1d_array_in_place*(arr: PyObject, scalar: float64) {.nuwa_export.} =
  ## Test helper: Multiply array in-place
  var npArr = asNumpyArrayWrite(arr, float64)

  for val in mitems(npArr):
    val = val * scalar

proc test_2d_matrix_multiply*(a: PyObject, b: PyObject): seq[seq[float64]] {.nuwa_export.} =
  ## Test helper: Matrix multiplication for 2D arrays
  var matA = asStridedArray(a, float64)
  var matB = asStridedArray(b, float64)

  let rowsA = matA.shape[0]
  let colsA = matA.shape[1]
  let colsB = matB.shape[1]

  if matB.shape[0] != colsA:
    raise newException(ValueError, "Matrix dimensions mismatch")

  result = newSeq[seq[float64]](rowsA)
  for i in 0..<rowsA:
    result[i] = newSeq[float64](colsB)
    for j in 0..<colsB:
      var sum = 0.0
      for k in 0..<colsA:
        sum += matA[i, k] * matB[k, j]
      result[i][j] = sum

proc test_array_properties*(arr: PyObject, writable: bool): tuple[
  len: int,
  ndim: int,
  shape: seq[int],
  is_contiguous: bool
] {.nuwa_export.} =
  ## Test helper: Return array properties
  ## Useful for verifying that shape/strides are extracted correctly
  if writable:
    var npArr = asNumpyArrayWrite(arr, float64)
    return (
      len: npArr.len,
      ndim: npArr.ndim,
      shape: npArr.shape,
      is_contiguous: npArr.isContiguous
    )
  else:
    var npArr = asNumpyArray(arr, float64)
    return (
      len: npArr.len,
      ndim: npArr.ndim,
      shape: npArr.shape,
      is_contiguous: npArr.isContiguous
    )

proc test_indexing_2d*(arr: PyObject): seq[seq[float64]] {.nuwa_export.} =
  ## Test helper: Extract diagonal elements from 2D array
  ## Tests multi-dimensional indexing
  var mat = asStridedArray(arr, float64)

  let rows = mat.shape[0]
  let cols = mat.shape[1]
  let size = min(rows, cols)

  result = newSeq[seq[float64]](size)
  for i in 0..<size:
    result[i] = @[mat[i, i]]

proc test_iterator_items*(arr: PyObject): seq[int64] {.nu_export.} =
  ## Test helper: Test iteration via items iterator
  ## Verifies that the iterator works correctly
  var npArr = asNumpyArray(arr, int64)

  result = newSeq[int64](0)
  for val in items(npArr):
    result.add(val)

proc test_iterator_pairs*(arr: PyObject): seq[tuple[idx: int, val: int64]] {.nuwa_export.} =
  ## Test helper: Test iteration via pairs iterator
  var npArr = asNumpyArray(arr, int64)

  result = newSeq[tuple[idx: int, val: int64]](0)
  for idx, val in pairs(npArr):
    result.add((idx[0], val))

proc test_reshape_copy*(arr: PyObject): seq[int64] {.nuwa_export.} =
  ## Test helper: Convert numpy array to Nim sequence
  ## Tests toSeq helper function
  var npArr = asNumpyArray(arr, int64)
  return toSeq(npArr)

proc test_dtype_validation*(arr: PyObject, expected_type: string): bool {.nuwa_export.} =
  ## Test helper: Verify dtype validation
  ## Returns true if array can be wrapped with specified type
  try:
    when defined(UseInt64):
      discard asNumpyArray(arr, int64)
    else:
      discard asNumpyArray(arr, float64)
    return true
  except:
    return false
