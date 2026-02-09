## =============================================================================
## Tests for NumPy Array Wrappers
## =============================================================================
## This test suite verifies the RAII-based numpy wrapper functionality
## including buffer management, indexing, iteration, and GIL release.
##
## Note: We use 'include' instead of 'import' for nuwa_sdk because templates
## need to be in the same compilation unit to be expanded properly.

import unittest
import nimpy
import std/sequtils
include nuwa_sdk

suite "Numpy Array Wrappers - Basic Operations":
  test "1D read-only array access":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3, 4, 5], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    check npArr.len == 5
    check npArr.ndim == 1
    check npArr.shape == @[5]
    check npArr.size == 5
    check npArr.isContiguous == true

    # Test indexing
    check npArr[0] == 1'i64
    check npArr[4] == 5'i64

  test "1D float array access":
    let np = pyImport("numpy")
    let arr = np.array([1.5, 2.5, 3.5], dtype="float64")

    let npArr = asNumpyArray(arr, float64)
    check npArr.len == 3
    check npArr[0] == 1.5
    check npArr[2] == 3.5

  test "1D read-only array iteration":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3, 4, 5], dtype="int64")

    let npArr = asNumpyArray(arr, int64)

    var sum = 0'i64
    for val in items(npArr):
      sum += val

    check sum == 15'i64

  test "1D read-only array pairs iteration":
    let np = pyImport("numpy")
    let arr = np.array([10, 20, 30], dtype="int64")

    let npArr = asNumpyArray(arr, int64)

    var count = 0
    for idx, val in pairs(npArr):
      check idx.len == 1
      check idx[0] == count
      check val == int64((count + 1) * 10)
      inc count

    check count == 3

suite "Numpy Array Wrappers - Mutable Arrays":
  test "1D mutable array creation and modification":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3, 4, 5], dtype="int64")

    let npArr = asNumpyArrayWrite(arr, int64)
    check npArr.len == 5
    check npArr.isContiguous == true

    # Modify elements
    npArr[0] = 10'i64
    npArr[4] = 50'i64

    check npArr[0] == 10'i64
    check npArr[4] == 50'i64

  test "1D mutable array iteration with mitems":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3, 4, 5], dtype="float64")

    let npArr = asNumpyArrayWrite(arr, float64)

    # Double all values
    for val in mitems(npArr):
      val = val * 2.0

    check npArr[0] == 2.0
    check npArr[2] == 6.0
    check npArr[4] == 10.0

suite "Numpy Array Wrappers - Multi-dimensional":
  test "2D array shape and dimensions":
    let np = pyImport("numpy")
    let mat = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")

    let npMat = asStridedArray(mat, int64)
    check npMat.ndim == 2
    check npMat.shape == @[2, 3]
    check npMat.size == 6

  test "2D array indexing":
    let np = pyImport("numpy")
    let mat = np.array([[1, 2, 3], [4, 5, 6]], dtype="float64")

    let npMat = asStridedArray(mat, float64)

    check npMat[0, 0] == 1.0
    check npMat[0, 2] == 3.0
    check npMat[1, 1] == 5.0
    check npMat[1, 2] == 6.0

  test "2D mutable array indexing":
    let np = pyImport("numpy")
    let mat = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")

    let npMat = asNumpyArrayWrite(mat, int64)

    # Modify specific elements
    npMat[0, 0] = 100'i64
    npMat[1, 2] = 600'i64

    check npMat[0, 0] == 100'i64
    check npMat[0, 1] == 2'i64
    check npMat[1, 2] == 600'i64

suite "Numpy Array Wrappers - RAII Cleanup":
  test "RAII cleanup on normal scope exit":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3], dtype="int64")

    block:
      let npArr = asNumpyArray(arr, int64)
      check npArr.len == 3
      # Buffer should be released automatically here

    # If we got here without segfault, RAII worked

  test "RAII cleanup on exception":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3], dtype="int64")

    try:
      let npArr = asNumpyArray(arr, int64)
      check npArr.len == 3
      raise newException(ValueError, "test exception")
    except ValueError:
      discard  # Buffer should be released here

    # If we got here without segfault, RAII worked on exception

suite "Numpy Array Wrappers - Type Conversion":
  test "toSeq conversion":
    let np = pyImport("numpy")
    let arr = np.array([10, 20, 30], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    let seq = toSeq(npArr)

    check seq.len == 3
    check seq[0] == 10'i64
    check seq[2] == 30'i64

  test "toOpenArray read-only view":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3, 4], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    let view = toOpenArray(npArr)

    check sum(view) == 10'i64

  test "toOpenArray writable view":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3], dtype="int64")

    let npArr = asNumpyArrayWrite(arr, int64)
    var view = toOpenArray(npArr)

    proc scale(a: var openArray[int64]) =
      for i in 0..<a.len:
        a[i] = a[i] * 2

    scale(view)

    check npArr[0] == 2'i64
    check npArr[1] == 4'i64
    check npArr[2] == 6'i64

suite "Numpy Array Wrappers - Utility Functions":
  test "isContiguous check":
    let np = pyImport("numpy")

    let arr1d = np.array([1, 2, 3], dtype="int64")
    let npArr1d = asNumpyArray(arr1d, int64)
    check npArr1d.isContiguous == true

    let mat = np.array([[1, 2], [3, 4]], dtype="int64")
    let npMat = asStridedArray(mat, int64)
    # 2D arrays created with np.array are typically contiguous
    # but we're forcing strided mode
    check npMat.isContiguous == false

suite "Numpy Array Wrappers - withNogil Integration":
  test "GIL release with data pointer":
    let np = pyImport("numpy")
    # Create a simple array for testing
    let data_list = newSeq[int](100)
    for i in 0..<100:
      data_list[i] = i
    let arr = np.array(data_list, dtype="float64")

    let npArr = asNumpyArray(arr, float64)
    let n = npArr.len

    var sum = 0.0

    # Get pointer before GIL release
    let data = npArr.data

    withNogil:
      # Pure Nim computation without GIL
      for i in 0..<n:
        sum += data[i]

    check abs(sum - 4950.0) < 0.01

suite "Numpy Array Wrappers - Edge Cases":
  test "Empty array":
    let np = pyImport("numpy")
    let arr = np.array([], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    check npArr.len == 0

  test "Single element array":
    let np = pyImport("numpy")
    let arr = np.array([42], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    check npArr.len == 1
    check npArr[0] == 42'i64

  test "Large array iteration":
    let np = pyImport("numpy")
    # Use a smaller large array for testing
    let data_list = newSeq[int](1000)
    for i in 0..<1000:
      data_list[i] = i
    let arr = np.array(data_list, dtype="int64")

    let npArr = asNumpyArray(arr, int64)

    var count = 0
    for val in items(npArr):
      inc count

    check count == 1000

suite "Numpy Array Wrappers - Different Types":
  test "int32 array":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3], dtype="int32")

    let npArr = asNumpyArray(arr, int32)
    check npArr.len == 3
    check npArr[0] == 1'i32

  test "float32 array":
    let np = pyImport("numpy")
    let arr = np.array([1.5, 2.5, 3.5], dtype="float32")

    let npArr = asNumpyArray(arr, float32)
    check npArr.len == 3
    check npArr[0] == 1.5'f32
