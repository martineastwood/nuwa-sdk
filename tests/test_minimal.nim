import unittest
import nimpy
include nuwa_sdk

suite "Minimal Numpy Tests":
  test "Basic array access":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    check npArr.len == 3
    check npArr[0] == 1'i64
    check npArr[2] == 3'i64

  test "Iteration works":
    let np = pyImport("numpy")
    let arr = np.array([1, 2, 3, 4, 5], dtype="int64")

    let npArr = asNumpyArray(arr, int64)
    var sum = 0'i64
    for val in items(npArr):
      sum += val

    check sum == 15'i64
