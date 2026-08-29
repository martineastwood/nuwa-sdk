## =============================================================================
## NumPy Array Wrappers for nuwa-sdk
## =============================================================================
## This module provides ergonomic, RAII-based wrappers for numpy arrays that
## build on nimpy's buffer protocol support. The wrappers provide automatic
## cleanup, separate contiguous/strided handling, and an intuitive API.
##
## Key Features:
## - RAII cleanup (automatic resource management)
## - Zero-copy access to numpy arrays
## - Multi-dimensional indexing with stride calculation
## - Type validation and bounds checking (debug mode)
## - Integration with withNogil for GIL release
## - Iterators for convenient data access
##
## Example:
##   proc sumArray(arr: PyObject): int64 {.nuwa_export.} =
##     let npArr = arr.asNumpyArray[int64]()
##     defer: npArr.close()
##
##     result = 0
##     for val in npArr:
##       result += val
##
##   # With GIL release:
##   proc sumArrayFast(arr: PyObject): int64 {.nuwa_export.} =
##     let npArr = arr.asNumpyArray[int64]()
##     defer: npArr.close()
##
##     let n = npArr.len
##     withNogil:
##       var sum = 0'i64
##       let data = npArr.data
##       for i in 0..<n:
##         sum += data[i]
##       return sum

import nimpy
import nimpy/raw_buffers
import nimpy/py_types
import std/strutils

# =============================================================================
# Type Definitions
# =============================================================================

type
  NumpyError* = object of ValueError
    ## Base error type for numpy-related errors

  LayoutError* = object of NumpyError
    ## Error raised when array layout doesn't match requirements

  DimensionError* = object of NumpyError
    ## Error raised when array dimensions don't match expectations

  TypeError* = object of NumpyError
    ## Error raised when array dtype doesn't match expected type

  NumpyArrayRead*[T] = object
    ## Read-only numpy array wrapper (any dimension)
    ## Provides RAII cleanup and multi-dimensional indexing
    buf*: RawPyBuffer
    owner*: PyObject            ## Keep reference to Python object
    shape*: seq[int]            ## Array dimensions
    strides*: seq[int]          ## Byte strides for each dimension
    itemSize*: int              ## Element size in bytes
    when defined(debug):
      initialized*: bool        ## Track initialization state
    isContiguous*: bool
    contiguousData*: ptr UncheckedArray[T]
    contiguousLen*: int

  NumpyArrayWrite*[T] = object
    ## Mutable numpy array wrapper (any dimension)
    ## Provides RAII cleanup and multi-dimensional indexing
    buf*: RawPyBuffer
    owner*: PyObject            ## Keep reference to Python object
    shape*: seq[int]            ## Array dimensions
    strides*: seq[int]          ## Byte strides for each dimension
    itemSize*: int              ## Element size in bytes
    when defined(debug):
      initialized*: bool        ## Track initialization state
    isContiguous*: bool
    contiguousData*: ptr UncheckedArray[T]
    contiguousLen*: int

# =============================================================================
# Private Helper Procedures
# =============================================================================

proc getDtypeCode[T](): char =
  ## Get numpy dtype code for Nim type T
  ## Compile-time error for unsupported types
  when T is int8:
    return 'b'
  elif T is int16:
    return 'h'
  elif T is int32:
    return 'i'
  elif T is int64:
    return 'q'
  elif T is uint8:
    return 'B'
  elif T is uint16:
    return 'H'
  elif T is uint32:
    return 'I'
  elif T is uint64:
    return 'Q'
  elif T is float32:
    return 'f'
  elif T is float64:
    return 'd'
  elif T is bool:
    return '?'
  else:
    {.error: "Unsupported type for numpy array: " & $T &
            ". Supported types: int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, bool.".}

proc parseFormat(
  format: string,
  endian: var char,
  code: var char,
  size: var int,
  hasSize: var bool
): bool =
  ## Parse a PEP 3118-style format string with optional endianness.
  ## Returns false if the format is not a simple scalar code.
  if format.len == 0:
    return false
  var i = 0
  endian = '@'
  if format[0] in {'@', '=', '<', '>', '!', '|'}:
    endian = format[0]
    i.inc
  if i >= format.len:
    return false
  code = format[i]
  i.inc
  size = 0
  hasSize = false
  while i < format.len and format[i].isDigit:
    size = size * 10 + (ord(format[i]) - ord('0'))
    hasSize = true
    i.inc
  if i != format.len:
    return false
  return true

proc codeSize(code: char): int =
  ## Map format code to element size (in bytes) when size is not explicit.
  case code
  of 'b', 'B', '?':
    return 1
  of 'h', 'H':
    return sizeof(cshort)
  of 'i', 'I':
    return sizeof(cint)
  of 'l', 'L':
    return sizeof(clong)
  of 'q', 'Q':
    return sizeof(clonglong)
  of 'f':
    return sizeof(cfloat)
  of 'd':
    return sizeof(cdouble)
  of 'g':
    return sizeof(clongdouble)
  else:
    return 0

proc isSignedIntCode(code: char): bool =
  code in {'b', 'h', 'i', 'l', 'q'}

proc isUnsignedIntCode(code: char): bool =
  code in {'B', 'H', 'I', 'L', 'Q'}

proc isFloatCode(code: char): bool =
  code in {'f', 'd', 'g'}

proc isNativeEndian(endian: char): bool =
  case endian
  of '@', '=', '|':
    true
  of '<':
    cpuEndian == littleEndian
  of '>', '!':
    cpuEndian == bigEndian
  else:
    false

proc validateDtype[T](buf: RawPyBuffer) =
  ## Validate that buffer format matches expected dtype
  ## Note: getDtypeCode[T]() will produce a compile-time error for unsupported types
  let expectedCode = getDtypeCode[T]()

  let actual = if buf.format.isNil: "" else: $buf.format
  if actual.len == 0:
    raise newException(TypeError,
      "Array dtype mismatch: expected '" & $expectedCode & "' but buffer has no format")

  var endian: char
  var code: char
  var size: int
  var hasSize: bool
  let parsed = parseFormat(actual, endian, code, size, hasSize)
  if not parsed:
    # Fallback for simple one-character formats
    if actual.len == 1 and actual[0] == expectedCode:
      return
    if actual.len == 2 and actual[0] in {'@', '=', '<', '>', '!', '|'} and actual[1] == expectedCode:
      return
    raise newException(TypeError,
      "Array dtype mismatch: expected '" & $expectedCode & "' but got '" & actual & "'")

  let actualSize = if hasSize: size else: codeSize(code)
  let expectedSize = sizeof(T)
  if actualSize == 0 or actualSize != expectedSize:
    raise newException(TypeError,
      "Array dtype mismatch: expected size " & $expectedSize & " but got '" & actual & "'")
  if expectedSize > 1 and not isNativeEndian(endian):
    raise newException(TypeError,
      "Array dtype mismatch: expected native-endian '" & $expectedCode & "' but got '" & actual & "'")

  when T is bool:
    if code != '?':
      raise newException(TypeError,
        "Array dtype mismatch: expected '?', got '" & actual & "'")
  elif T is float32 or T is float64:
    if not isFloatCode(code):
      raise newException(TypeError,
        "Array dtype mismatch: expected float, got '" & actual & "'")
  elif T is int8 or T is int16 or T is int32 or T is int64:
    if not isSignedIntCode(code):
      raise newException(TypeError,
        "Array dtype mismatch: expected signed int, got '" & actual & "'")
  elif T is uint8 or T is uint16 or T is uint32 or T is uint64:
    if not isUnsignedIntCode(code):
      raise newException(TypeError,
        "Array dtype mismatch: expected unsigned int, got '" & actual & "'")

proc computeShape(buf: RawPyBuffer): seq[int] =
  ## Extract shape from buffer
  if buf.ndim <= 0:
    return @[]

  result = newSeq[int](buf.ndim)
  let shapePtr = cast[ptr UncheckedArray[Py_ssize_t]](buf.shape)
  for i in 0..<buf.ndim:
    result[i] = int(shapePtr[i])

proc itemSizeFromFormat(format: string): int =
  ## Best-effort element size from format string.
  var endian: char
  var code: char
  var size: int
  var hasSize: bool
  if parseFormat(format, endian, code, size, hasSize):
    if hasSize:
      return size
    return codeSize(code)
  return 0

proc getItemSize[T](buf: RawPyBuffer): int =
  ## Best-effort element size in bytes.
  when compiles(buf.itemsize):
    if buf.itemsize > 0:
      return int(buf.itemsize)

  let fmt = if buf.format.isNil: "" else: $buf.format
  let sizeFromFmt = itemSizeFromFormat(fmt)
  if sizeFromFmt > 0:
    return sizeFromFmt

  if buf.strides != nil and buf.ndim > 0:
    let stridesPtr = cast[ptr UncheckedArray[Py_ssize_t]](buf.strides)
    let lastStride = int(stridesPtr[buf.ndim - 1])
    if lastStride != 0:
      return abs(lastStride)

  return sizeof(T)

proc computeStrides(buf: RawPyBuffer, shape: seq[int], itemSize: int): seq[int] =
  ## Extract or compute strides from buffer
  if buf.ndim <= 0:
    return @[]

  if buf.strides.isNil:
    result = newSeq[int](buf.ndim)
    var stride = itemSize
    for i in countdown(buf.ndim - 1, 0):
      result[i] = stride
      let dim = if shape[i] > 0: shape[i] else: 1
      stride *= dim
    return

  result = newSeq[int](buf.ndim)
  let stridesPtr = cast[ptr UncheckedArray[Py_ssize_t]](buf.strides)
  for i in 0..<buf.ndim:
    result[i] = int(stridesPtr[i])

proc isCContiguous(shape: seq[int], strides: seq[int], itemSize: int): bool =
  ## Check if array is C-contiguous (row-major)
  ## Size-1 dimensions (singletons) are stride-flexible per NumPy semantics
  if shape.len <= 1:
    if shape.len == 0:
      return true
    return strides.len == 0 or strides[0] == itemSize

  if strides.len != shape.len:
    return false

  var expectedStride = itemSize
  for i in countdown(shape.len - 1, 0):
    # Size-1 dimensions can have any stride (NumPy compatibility)
    if shape[i] != 1 and strides[i] != expectedStride:
      return false
    if shape[i] > 1:
      expectedStride *= shape[i]
  return true

proc isFortranContiguous(shape: seq[int], strides: seq[int], itemSize: int): bool =
  ## Check if array is Fortran-contiguous (column-major)
  ## Size-1 dimensions (singletons) are stride-flexible per NumPy semantics
  if shape.len <= 1:
    if shape.len == 0:
      return true
    return strides.len == 0 or strides[0] == itemSize

  if strides.len != shape.len:
    return false

  var expectedStride = itemSize
  for i in 0..<shape.len:
    # Size-1 dimensions can have any stride (NumPy compatibility)
    if shape[i] != 1 and strides[i] != expectedStride:
      return false
    if shape[i] > 1:
      expectedStride *= shape[i]
  return true

proc getStrideElements(stride: int, itemSize: int, dimIdx: int): int =
  ## Convert byte stride to element stride with validation
  ## Strides must be aligned to element size for proper indexing
  when defined(debug):
    if itemSize > 0 and (stride mod itemSize != 0):
      raise newException(LayoutError,
        "Stride " & $stride & " for dimension " & $dimIdx &
        " is not aligned to element size " & $itemSize)

  if itemSize > 0:
    result = stride div itemSize
  else:
    result = 0

# =============================================================================
# RAII Cleanup (Destructors)
# =============================================================================

proc `=destroy`*[T](arr: NumpyArrayRead[T]) =
  ## Automatic cleanup when read-only array goes out of scope
  ## Note: arr is passed by value, so we can't modify the original.
  ## The object will be gone after this proc returns.
  if arr.buf.buf != nil:
    var buf = arr.buf
    release(buf)

proc `=destroy`*[T](arr: NumpyArrayWrite[T]) =
  ## Automatic cleanup when writable array goes out of scope
  ## Note: arr is passed by value, so we can't modify the original.
  ## The object will be gone after this proc returns.
  if arr.buf.buf != nil:
    var buf = arr.buf
    release(buf)

proc `=copy`*[T](dest: var NumpyArrayRead[T], source: NumpyArrayRead[T]) {.error: "NumpyArrayRead cannot be copied (would cause double-free). Use `var arr2 = arr1` to move, or pass by ref.".}

proc `=copy`*[T](dest: var NumpyArrayWrite[T], source: NumpyArrayWrite[T]) {.error: "NumpyArrayWrite cannot be copied (would cause double-free). Use `var arr2 = arr1` to move, or pass by ref.".}

proc `=sink`*[T](dest: var NumpyArrayRead[T], source: NumpyArrayRead[T]) =
  ## Move semantics - transfer ownership from source to dest
  ## Source is an expiring value and won't be destroyed separately
  if dest.buf.buf != nil:
    release(dest.buf)

  # Move all fields - source's destructor won't be called
  dest.buf = source.buf
  dest.owner = source.owner
  dest.shape = source.shape
  dest.strides = source.strides
  dest.itemSize = source.itemSize
  dest.isContiguous = source.isContiguous
  dest.contiguousData = source.contiguousData
  dest.contiguousLen = source.contiguousLen
  when defined(debug):
    dest.initialized = source.initialized

proc `=sink`*[T](dest: var NumpyArrayWrite[T], source: NumpyArrayWrite[T]) =
  ## Move semantics - transfer ownership from source to dest
  ## Source is an expiring value and won't be destroyed separately
  if dest.buf.buf != nil:
    release(dest.buf)

  # Move all fields - source's destructor won't be called
  dest.buf = source.buf
  dest.owner = source.owner
  dest.shape = source.shape
  dest.strides = source.strides
  dest.itemSize = source.itemSize
  dest.isContiguous = source.isContiguous
  dest.contiguousData = source.contiguousData
  dest.contiguousLen = source.contiguousLen
  when defined(debug):
    dest.initialized = source.initialized

proc close*[T](arr: var NumpyArrayRead[T]) {.inline.} =
  ## Explicit cleanup (optional, RAII handles it automatically)
  if arr.buf.buf != nil:
    release(arr.buf)
    arr.buf.buf = nil
  arr.contiguousData = nil
  arr.contiguousLen = 0
  when defined(debug):
    arr.initialized = false

proc close*[T](arr: var NumpyArrayWrite[T]) {.inline.} =
  ## Explicit cleanup (optional, RAII handles it automatically)
  if arr.buf.buf != nil:
    release(arr.buf)
    arr.buf.buf = nil
  arr.contiguousData = nil
  arr.contiguousLen = 0
  when defined(debug):
    arr.initialized = false

# =============================================================================
# Smart Constructor - tries contiguous fast path
# =============================================================================

proc asNumpyArray*[T](arr: PyObject, writable: bool = false): NumpyArrayRead[T] =
  ## Convert Python object to read-only numpy array wrapper
  ## Tries contiguous fast path first, falls back to strided access
  ##
  ## Parameters:
  ##   arr - Python object (numpy array or buffer protocol object)
  ##   writable - Whether to require a writable buffer (still returned as read-only)
  ##
  ## Returns:
  ##   NumpyArrayRead[T] wrapper with automatic cleanup
  ##
  ## Example:
  ##   let npArr = pyArray.asNumpyArray[int64]()
  ##   defer: npArr.close()
  ##
  ##   for val in npArr:
  ##     echo val

  let mode = if writable:
    PyBUF_WRITE or PyBUF_STRIDED or PyBUF_FORMAT
  else:
    PyBUF_READ or PyBUF_STRIDED_RO or PyBUF_FORMAT

  var buf: RawPyBuffer
  getBuffer(arr, buf, mode.cint)

  try:
    # Validate dtype
    validateDtype[T](buf)

    let shape = computeShape(buf)
    let itemSize = getItemSize[T](buf)
    let strides = computeStrides(buf, shape, itemSize)
    let contiguous = isCContiguous(shape, strides, itemSize)

    result.buf = buf
    result.owner = arr
    result.shape = shape
    result.strides = strides
    result.itemSize = itemSize
    result.isContiguous = contiguous
    if contiguous:
      result.contiguousData = cast[ptr UncheckedArray[T]](buf.buf)
      result.contiguousLen = if itemSize > 0: buf.len div itemSize else: 0

    when defined(debug):
      result.initialized = true
  except:
    if buf.buf != nil:
      release(buf)
    raise

proc asNumpyArrayWrite*[T](arr: PyObject): NumpyArrayWrite[T] =
  ## Convert Python object to writable numpy array wrapper
  ## Tries contiguous fast path first, falls back to strided access
  ##
  ## Parameters:
  ##   arr - Python object (must be writable numpy array)
  ##
  ## Returns:
  ##   NumpyArrayWrite[T] wrapper with automatic cleanup
  ##
  ## Example:
  ##   let npArr = pyArray.asNumpyArrayWrite[float64]()
  ##   defer: npArr.close()
  ##
  ##   for i in 0..<npArr.len:
  ##     npArr[i] = npArr[i] * 2.0

  let mode = PyBUF_WRITE or PyBUF_STRIDED or PyBUF_FORMAT

  var buf: RawPyBuffer
  getBuffer(arr, buf, mode.cint)

  try:
    # Validate dtype
    validateDtype[T](buf)

    let shape = computeShape(buf)
    let itemSize = getItemSize[T](buf)
    let strides = computeStrides(buf, shape, itemSize)
    let contiguous = isCContiguous(shape, strides, itemSize)

    result.buf = buf
    result.owner = arr
    result.shape = shape
    result.strides = strides
    result.itemSize = itemSize
    result.isContiguous = contiguous
    if contiguous:
      result.contiguousData = cast[ptr UncheckedArray[T]](buf.buf)
      result.contiguousLen = if itemSize > 0: buf.len div itemSize else: 0

    when defined(debug):
      result.initialized = true
  except:
    if buf.buf != nil:
      release(buf)
    raise

proc asStridedArray*[T](arr: PyObject, writable: static bool = false): auto =
  ## Request a strided-capable buffer view (works for contiguous arrays too)
  ##
  ## Use this when you know the array is multi-dimensional or want to
  ## explicitly handle strided access patterns.
  ##
  ## Example:
  ##   let mat = npArray.asStridedArray[float64]()
  ##   defer: mat.close()
  ##
  ##   echo mat[0, 0]  # Multi-dimensional indexing

  template initArray(resultVar: typed): untyped =
    let mode = when writable:
      PyBUF_WRITE or PyBUF_STRIDED or PyBUF_FORMAT
    else:
      PyBUF_READ or PyBUF_STRIDED_RO or PyBUF_FORMAT

    var buf: RawPyBuffer
    getBuffer(arr, buf, mode.cint)
    try:
      validateDtype[T](buf)
      let shape = computeShape(buf)
      let itemSize = getItemSize[T](buf)
      let strides = computeStrides(buf, shape, itemSize)
      let contiguous = isCContiguous(shape, strides, itemSize)

      resultVar.buf = buf
      resultVar.owner = arr
      resultVar.shape = shape
      resultVar.strides = strides
      resultVar.itemSize = itemSize
      resultVar.isContiguous = contiguous
      if contiguous:
        resultVar.contiguousData = cast[ptr UncheckedArray[T]](buf.buf)
        resultVar.contiguousLen = if itemSize > 0: buf.len div itemSize else: 0

      when defined(debug):
        resultVar.initialized = true
    except:
      if buf.buf != nil:
        release(buf)
      raise

  when writable:
    var result: NumpyArrayWrite[T]
    initArray(result)
    return result
  else:
    var result: NumpyArrayRead[T]
    initArray(result)
    return result

template asNumpyArray*(arr: PyObject, T: typedesc): NumpyArrayRead[T] =
  ## Backward-compatible typedesc overload.
  asNumpyArray[T](arr)

template asNumpyArrayWrite*(arr: PyObject, T: typedesc): NumpyArrayWrite[T] =
  ## Backward-compatible typedesc overload.
  asNumpyArrayWrite[T](arr)

template asStridedArray*(arr: PyObject, T: typedesc): NumpyArrayRead[T] =
  ## Backward-compatible typedesc overload.
  asStridedArray[T](arr)

# =============================================================================
# Data Access API - Properties
# =============================================================================

proc data*[T](arr: NumpyArrayRead[T]): ptr UncheckedArray[T] {.inline.} =
  ## Get raw pointer to array data (for contiguous arrays or use with GIL release)
  ## Only valid for contiguous arrays - raises LayoutError for strided arrays
  if not arr.isContiguous:
    raise newException(LayoutError, "Cannot get raw pointer for non-contiguous array")
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  arr.contiguousData

proc data*[T](arr: NumpyArrayWrite[T]): ptr UncheckedArray[T] {.inline.} =
  ## Get raw pointer to array data (for contiguous arrays or use with GIL release)
  ## Only valid for contiguous arrays - raises LayoutError for strided arrays
  if not arr.isContiguous:
    raise newException(LayoutError, "Cannot get raw pointer for non-contiguous array")
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  arr.contiguousData

proc len*[T](arr: NumpyArrayRead[T]): int {.inline.} =
  ## Get total number of elements in the array
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  if arr.isContiguous:
    return arr.contiguousLen
  else:
    result = 1
    for dim in arr.shape:
      result *= dim

proc len*[T](arr: NumpyArrayWrite[T]): int {.inline.} =
  ## Get total number of elements in the array
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  if arr.isContiguous:
    return arr.contiguousLen
  else:
    result = 1
    for dim in arr.shape:
      result *= dim

proc shape*[T](arr: NumpyArrayRead[T]): seq[int] {.inline.} =
  ## Get array dimensions as a sequence
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
  arr.shape

proc shape*[T](arr: NumpyArrayWrite[T]): seq[int] {.inline.} =
  ## Get array dimensions as a sequence
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
  arr.shape

proc ndim*[T](arr: NumpyArrayRead[T]): int {.inline.} =
  ## Get number of dimensions
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
  arr.shape.len

proc ndim*[T](arr: NumpyArrayWrite[T]): int {.inline.} =
  ## Get number of dimensions
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
  arr.shape.len

proc size*[T](arr: NumpyArrayRead[T]): int {.inline.} =
  ## Get total element count (alias for len)
  arr.len

proc size*[T](arr: NumpyArrayWrite[T]): int {.inline.} =
  ## Get total element count (alias for len)
  arr.len

# =============================================================================
# 1D Contiguous Array Indexing
# =============================================================================

proc `[]`*[T](arr: NumpyArrayRead[T], i: int): T {.inline.} =
  ## Safe indexing for 1D contiguous arrays
  ## Contiguity and initialization checks are always performed (release safety)
  ## Bounds checking is debug-only (use with caution in release)
  if not arr.isContiguous:
    raise newException(LayoutError, "Use multi-dimensional indexing for non-contiguous arrays")
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
    if i < 0 or i >= arr.contiguousLen:
      raise newException(IndexError, "Index " & $i & " out of bounds [0, " & $arr.contiguousLen & ")")

  arr.contiguousData[i]

proc `[]`*[T](arr: NumpyArrayWrite[T], i: int): T {.inline.} =
  ## Safe indexing for 1D contiguous arrays
  ## Contiguity and initialization checks are always performed (release safety)
  ## Bounds checking is debug-only (use with caution in release)
  if not arr.isContiguous:
    raise newException(LayoutError, "Use multi-dimensional indexing for non-contiguous arrays")
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
    if i < 0 or i >= arr.contiguousLen:
      raise newException(IndexError, "Index " & $i & " out of bounds [0, " & $arr.contiguousLen & ")")

  arr.contiguousData[i]

proc `[]=`*[T](arr: NumpyArrayWrite[T], i: int, val: T) {.inline.} =
  ## Safe element assignment for 1D contiguous arrays
  ## Contiguity and initialization checks are always performed (release safety)
  ## Bounds checking is debug-only (use with caution in release)
  if not arr.isContiguous:
    raise newException(LayoutError, "Use multi-dimensional indexing for non-contiguous arrays")
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
    if i < 0 or i >= arr.contiguousLen:
      raise newException(IndexError, "Index " & $i & " out of bounds [0, " & $arr.contiguousLen & ")")

  arr.contiguousData[i] = val

# =============================================================================
# Multi-Dimensional Strided Array Indexing
# =============================================================================

proc `[]`*[T](arr: NumpyArrayRead[T], indices: varargs[int]): T {.inline.} =
  ## Multi-dimensional indexing for strided arrays
  ## Automatically calculates offset using strides
  ##
  ## Example:
  ##   let mat = npArray.asStridedArray[float64]()
  ##   echo mat[0, 0]  # First row, first column
  ##   echo mat[1, 2]  # Second row, third column

  let ndim = arr.shape.len

  # Dimension check is always performed (release safety)
  if ndim == 0 and indices.len != 0:
    raise newException(DimensionError, "Expected 0 indices for scalar array, got " & $indices.len)
  if ndim > 0 and indices.len != ndim:
    raise newException(DimensionError, "Expected " & $ndim & " indices, got " & $indices.len)

  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  # Calculate offset using strides (converted to element units)
  var offset = 0
  for i in 0..<ndim:
    let idx = indices[i]

    when defined(debug):
      if idx < 0 or idx >= arr.shape[i]:
        raise newException(IndexError, "Index " & $idx & " out of bounds for dimension " & $i & " [0, " & $arr.shape[i] & ")")

    let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
    offset += idx * strideElements

  let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
  return buf[offset]

proc `[]`*[T](arr: NumpyArrayWrite[T], indices: varargs[int]): T {.inline.} =
  ## Multi-dimensional indexing for strided arrays
  let ndim = arr.shape.len

  # Dimension check is always performed (release safety)
  if ndim == 0 and indices.len != 0:
    raise newException(DimensionError, "Expected 0 indices for scalar array, got " & $indices.len)
  if ndim > 0 and indices.len != ndim:
    raise newException(DimensionError, "Expected " & $ndim & " indices, got " & $indices.len)

  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  var offset = 0
  for i in 0..<ndim:
    let idx = indices[i]

    when defined(debug):
      if idx < 0 or idx >= arr.shape[i]:
        raise newException(IndexError, "Index " & $idx & " out of bounds for dimension " & $i & " [0, " & $arr.shape[i] & ")")

    let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
    offset += idx * strideElements

  let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
  return buf[offset]

proc `[]=`*[T](arr: NumpyArrayWrite[T], indices: varargs[int], val: T) {.inline.} =
  ## Multi-dimensional element assignment for strided arrays
  let ndim = arr.shape.len

  # Dimension check is always performed (release safety)
  if ndim == 0 and indices.len != 0:
    raise newException(DimensionError, "Expected 0 indices for scalar array, got " & $indices.len)
  if ndim > 0 and indices.len != ndim:
    raise newException(DimensionError, "Expected " & $ndim & " indices, got " & $indices.len)

  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  var offset = 0
  for i in 0..<ndim:
    let idx = indices[i]

    when defined(debug):
      if idx < 0 or idx >= arr.shape[i]:
        raise newException(IndexError, "Index " & $idx & " out of bounds for dimension " & $i & " [0, " & $arr.shape[i] & ")")

    let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
    offset += idx * strideElements

  let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
  buf[offset] = val

# =============================================================================
# Iterators
# =============================================================================

iterator items*[T](arr: NumpyArrayRead[T]): T =
  ## Iterate over elements in read-only array
  ## For contiguous arrays, this is a fast flat iteration
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  if arr.isContiguous:
    for i in 0..<arr.contiguousLen:
      yield arr.contiguousData[i]
  else:
    # Flat iteration over strided array
    let total = arr.size
    let ndim = arr.ndim
    var indices = newSeq[int](ndim)

    for _ in 0..<total:
      # Calculate offset for current element position
      var offset = 0
      for i in 0..<ndim:
        let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
        offset += indices[i] * strideElements

      let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
      yield buf[offset]

      # Increment indices (like odometer)
      var carry = 1
      for i in countdown(ndim - 1, 0):
        indices[i] += carry
        if indices[i] >= arr.shape[i]:
          indices[i] = 0
          carry = 1
        else:
          carry = 0
          break

iterator items*[T](arr: NumpyArrayWrite[T]): T =
  ## Iterate over elements in writable array (read-only view)
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  if arr.isContiguous:
    for i in 0..<arr.contiguousLen:
      yield arr.contiguousData[i]
  else:
    let total = arr.size
    let ndim = arr.ndim
    var indices = newSeq[int](ndim)

    for _ in 0..<total:
      # Calculate offset for current element position
      var offset = 0
      for i in 0..<ndim:
        let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
        offset += indices[i] * strideElements

      let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
      yield buf[offset]

      var carry = 1
      for i in countdown(ndim - 1, 0):
        indices[i] += carry
        if indices[i] >= arr.shape[i]:
          indices[i] = 0
          carry = 1
        else:
          carry = 0
          break

iterator mitems*[T](arr: NumpyArrayWrite[T]): var T =
  ## Iterate over mutable elements in writable array
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  if arr.isContiguous:
    for i in 0..<arr.contiguousLen:
      yield arr.contiguousData[i]
  else:
    let total = arr.size
    let ndim = arr.ndim
    var indices = newSeq[int](ndim)

    for _ in 0..<total:
      let offset = block:
        var off = 0
        for i in 0..<ndim:
          let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
          off += indices[i] * strideElements
        off

      let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
      yield buf[offset]

      var carry = 1
      for i in countdown(ndim - 1, 0):
        indices[i] += carry
        if indices[i] >= arr.shape[i]:
          indices[i] = 0
          carry = 1
        else:
          carry = 0
          break

iterator pairs*[T](arr: NumpyArrayRead[T]): tuple[idx: seq[int], val: T] =
  ## Iterate over elements with their multi-dimensional indices
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  let total = arr.size
  let ndim = arr.ndim
  if ndim == 1 and arr.isContiguous:
    for i in 0..<arr.contiguousLen:
      yield (@[i], arr.contiguousData[i])
  else:
    var indices = newSeq[int](ndim)

    for _ in 0..<total:
      # Calculate offset for current element position
      var offset = 0
      for i in 0..<ndim:
        let strideElements = getStrideElements(arr.strides[i], arr.itemSize, i)
        offset += indices[i] * strideElements

      let buf = cast[ptr UncheckedArray[T]](arr.buf.buf)
      var idxCopy = newSeq[int](indices.len)
      for i in 0..<indices.len:
        idxCopy[i] = indices[i]
      yield (idxCopy, buf[offset])

      var carry = 1
      for i in countdown(ndim - 1, 0):
        indices[i] += carry
        if indices[i] >= arr.shape[i]:
          indices[i] = 0
          carry = 1
        else:
          carry = 0
          break

# =============================================================================
# Utility Functions
# =============================================================================

template toOpenArray*[T](arr: NumpyArrayRead[T]): openArray[T] =
  ## Zero-copy view for 1D contiguous arrays compatible with std algorithms
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
  if not arr.isContiguous or arr.shape.len != 1:
    raise newException(LayoutError, "Cannot create openArray from non-1D contiguous array")
  block:
    if arr.contiguousLen == 0:
      toOpenArray(arr.contiguousData, 0, -1)
    else:
      toOpenArray(arr.contiguousData, 0, arr.contiguousLen - 1)

template toOpenArray*[T](arr: NumpyArrayWrite[T]): openArray[T] =
  ## Zero-copy view for 1D contiguous arrays compatible with std algorithms
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")
  if not arr.isContiguous or arr.shape.len != 1:
    raise newException(LayoutError, "Cannot create openArray from non-1D contiguous array")
  block:
    if arr.contiguousLen == 0:
      toOpenArray(arr.contiguousData, 0, -1)
    else:
      toOpenArray(arr.contiguousData, 0, arr.contiguousLen - 1)

proc toSeq*[T](arr: NumpyArrayRead[T]): seq[T] =
  ## Convert array to Nim sequence (copies data)
  when defined(debug):
    if not arr.initialized:
      raise newException(ValueError, "Array not initialized")

  result = newSeq[T](arr.len)
  var i = 0
  for val in arr:
    result[i] = val
    i.inc

proc isContiguous*[T](arr: NumpyArrayRead[T]): bool {.inline.} =
  ## Check if array is contiguous
  arr.isContiguous

proc isContiguous*[T](arr: NumpyArrayWrite[T]): bool {.inline.} =
  ## Check if array is contiguous
  arr.isContiguous

proc isFortranContiguous*[T](arr: NumpyArrayRead[T]): bool {.inline.} =
  ## Check if array is Fortran-contiguous (column-major)
  isFortranContiguous(arr.shape, arr.strides, arr.itemSize)

proc isFortranContiguous*[T](arr: NumpyArrayWrite[T]): bool {.inline.} =
  ## Check if array is Fortran-contiguous (column-major)
  isFortranContiguous(arr.shape, arr.strides, arr.itemSize)

proc `$`*[T](arr: NumpyArrayRead[T]): string =
  ## String representation for debugging
  when defined(debug):
    if not arr.initialized:
      return "NumpyArrayRead(uninitialized)"

  result = "NumpyArrayRead["
  if arr.isContiguous:
    result.add("contiguous, shape=" & $arr.shape)
  else:
    result.add("strided, shape=" & $arr.shape)
  result.add("]")

proc `$`*[T](arr: NumpyArrayWrite[T]): string =
  ## String representation for debugging
  when defined(debug):
    if not arr.initialized:
      return "NumpyArrayWrite(uninitialized)"

  result = "NumpyArrayWrite["
  if arr.isContiguous:
    result.add("contiguous, shape=" & $arr.shape)
  else:
    result.add("strided, shape=" & $arr.shape)
  result.add("]")
