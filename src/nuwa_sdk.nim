import macros, json, strutils, hashes
import nimpy
import nimpy/py_lib
import nimpy/raw_buffers
import std/dynlib

# =============================================================================
# GIL Management (Cython-style "with nogil" for Nim)
# =============================================================================

var
  PyEval_SaveThread_Fn*: proc(): pointer {.cdecl.}
  PyEval_RestoreThread_Fn*: proc(tstate: pointer) {.cdecl.}
  gilFunctionsInitialized = false

proc initGILFunctions*() =
  ## Load GIL management functions from Python library
  ## Called automatically during first use (lazy initialization)
  if not gilFunctionsInitialized:
    let libHandle = py_lib.pyLib.module
    if not libHandle.isNil:
      PyEval_SaveThread_Fn = cast[typeof(PyEval_SaveThread_Fn)](symAddr(libHandle, "PyEval_SaveThread"))
      PyEval_RestoreThread_Fn = cast[typeof(PyEval_RestoreThread_Fn)](symAddr(libHandle, "PyEval_RestoreThread"))
      gilFunctionsInitialized = true

template withNogil*(body: untyped) =
  ## Release the Python GIL, execute body, then reacquire the GIL
  ## This allows pure Nim code to run without Python interpreter interference
  ## Equivalent to Cython's "with nogil:" block or Py_BEGIN_ALLOW_THREADS in C
  block:
    # Ensure GIL functions are initialized before use
    if not gilFunctionsInitialized:
      initGILFunctions()

    let tstate = PyEval_SaveThread_Fn()
    try:
      body
    finally:
      PyEval_RestoreThread_Fn(tstate)

# =============================================================================
# Export Macro with Stub Generation
# =============================================================================

proc mapTypeToPython(nimType: string): string =
  ## Maps Nim types to Python type annotations
  case nimType
  of "int", "int64", "int32", "uint", "uint64", "uint32": return "int"
  of "float", "float64", "float32": return "float"
  of "string": return "str"
  of "bool": return "bool"
  of "void": return "None"
  of "auto": return "Any"
  # Handle sequences (list[T])
  elif nimType.startsWith("seq["):
    var inner = nimType[4 .. ^2] # Strip seq[ and ]
    return "list[" & mapTypeToPython(inner) & "]"
  # Handle arrays
  elif nimType.startsWith("array["):
    # array[N, T] - we'll simplify to list[T]
    let parts = nimType[6 .. ^2].split(",")
    if parts.len == 2:
      return "list[" & mapTypeToPython(parts[1].strip()) & "]"
    return "list[Any]"
  else:
    return "Any" # Fallback for unknown types

macro nuwa_export*(prc: untyped): untyped =
  ## Wraps nimpy's exportpy but also generates metadata for stub generation.
  ## If -d:nuwaStubDir=/tmp/path is set, writes JSON files there.
  ## Otherwise, prints "NUWA_STUB:..." to stdout.

  # 1. Validate input is a proc
  prc.expectKind(nnkProcDef)

  let funcName = prc.name.strVal
  var docString = ""
  var args: seq[JsonNode] = @[]
  var returnType = "None"

  # 2. Extract Docstring (collect all consecutive comment statements at start)
  let body = prc.body
  if body.kind == nnkStmtList:
    var docLines = newSeq[string]()
    for stmt in body:
      if stmt.kind == nnkCommentStmt:
        docLines.add(stmt.strVal.strip())
      elif stmt.kind == nnkEmpty:
        continue
      else:
        break # Stop at actual code
    docString = if docLines.len > 0: docLines.join("\n") else: ""

  # 3. Extract Parameters & Return Type
  let params = prc.params
  var firstParam = true

  for param in params:
    if firstParam:
      # First node in params is the Return Type
      if param.kind != nnkEmpty:
        returnType = mapTypeToPython(param.repr)
      firstParam = false
    else:
      # Remaining nodes are arguments (IdentDefs)
      # param structure: [name1, name2, ..., type, defaultVal]
      if param.len >= 2:
        let paramType = param[^2].repr # Type is second to last

        # Check for default value
        let hasDefault = param.len >= 3 and param[^1].kind != nnkEmpty

        let pyType = mapTypeToPython(paramType)

        # Iterate over names (in case of 'proc x(a, b: int)')
        for i in 0 ..< param.len - 2:
          let argName = param[i].strVal
          let argObj = %* {
            "name": argName,
            "type": pyType,
            "hasDefault": hasDefault
          }
          args.add(argObj)

  # 4. Construct JSON Payload
  let payload = %* {
    "name": funcName,
    "args": args,
    "returnType": returnType,
    "doc": docString
  }

  # Convert payload to string once to inject into quote block
  let payloadStr = $payload

  # 5. Generate Compile-Time Logger (The "Hook")
  # We use static: ... to execute this during compilation
  let logger = quote do:
    static:
      # Define the compile-time string flag.
      # Defaults to empty string if not passed by nuwa-build.
      const stubDir {.strdefine: "nuwaStubDir".}: string = ""
      let pStr = `payloadStr`

      if stubDir.len > 0:
        # File-based mode: Write specific JSON file
        # We hash the payload to ensure unique filenames for overloads
        var h = hash(pStr)
        let fname = stubDir & "/" & `funcName` & "_" & $h & ".json"
        writeFile(fname, pStr)
      else:
        # Fallback/Legacy mode: Print to stdout
        echo "NUWA_STUB:" & pStr

  # 6. Return standard nimpy export + our logger
  # Call nimpy's exportpy macro on the proc
  let exportpyCall = newCall(bindSym"exportpy", prc)
  result = quote do:
    `logger`
    `exportpyCall`

# =============================================================================
# NumPy Array Wrappers (RAII-based zero-copy access)
# =============================================================================
## This section provides ergonomic, RAII-based wrappers for numpy arrays.
## The wrappers provide automatic cleanup, multi-dimensional indexing, and
## integrate seamlessly with nuwa_export and withNogil.

type
  NumpyArrayRead*[T] = object
    ## Read-only numpy array wrapper (any dimension)
    ## Provides RAII cleanup and multi-dimensional indexing
    buf: RawPyBuffer
    owner*: PyObject
    shape*: seq[int]
    strides*: seq[int]

  NumpyArrayWrite*[T] = object
    ## Mutable numpy array wrapper (any dimension)
    buf: RawPyBuffer
    owner*: PyObject
    shape*: seq[int]
    strides*: seq[int]

proc `=destroy`*[T](arr: NumpyArrayRead[T]) =
  if arr.buf.buf != nil:
    var buf = arr.buf
    release(buf)

proc `=destroy`*[T](arr: NumpyArrayWrite[T]) =
  if arr.buf.buf != nil:
    var buf = arr.buf
    release(buf)

proc close*[T](arr: var NumpyArrayRead[T]) {.inline.} =
  `=destroy`(arr)

proc close*[T](arr: var NumpyArrayWrite[T]) {.inline.} =
  `=destroy`(arr)

template asNumpyArray*(arr: PyObject, T: typedesc): NumpyArrayRead[T] =
  ## Convert Python object to read-only numpy array wrapper
  ## Usage: let npArr = arr.asNumpyArray(int64)
  var result: NumpyArrayRead[T]
  var buf: RawPyBuffer
  let mode = cint(PyBUF_READ or PyBUF_CONTIG_RO)
  getBuffer(arr, buf, mode)
  result.buf = buf
  result.owner = arr

  # Extract shape
  if result.buf.ndim == 0:
    result.shape = @[result.buf.len div sizeof(T)]
  else:
    result.shape = newSeq[int](result.buf.ndim)
    let shapePtr = cast[ptr UncheckedArray[clong]](result.buf.shape)
    for i in 0..<result.buf.ndim:
      result.shape[i] = int(shapePtr[i])
  result

template asNumpyArrayWrite*(arr: PyObject, T: typedesc): NumpyArrayWrite[T] =
  ## Convert Python object to writable numpy array wrapper
  ## Usage: let npArr = arr.asNumpyArrayWrite(float64)
  var result: NumpyArrayWrite[T]
  var buf: RawPyBuffer
  let mode = cint(PyBUF_WRITE or PyBUF_CONTIG)
  getBuffer(arr, buf, mode)
  result.buf = buf
  result.owner = arr

  # Extract shape
  if result.buf.ndim == 0:
    result.shape = @[result.buf.len div sizeof(T)]
  else:
    result.shape = newSeq[int](result.buf.ndim)
    let shapePtr = cast[ptr UncheckedArray[clong]](result.buf.shape)
    for i in 0..<result.buf.ndim:
      result.shape[i] = int(shapePtr[i])
  result

template asStridedArray*(arr: PyObject, T: typedesc): NumpyArrayRead[T] =
  ## Force strided mode (for multi-dimensional or non-contiguous arrays)
  ## Usage: let mat = arr.asStridedArray(float64)
  var result: NumpyArrayRead[T]
  var buf: RawPyBuffer
  let mode = cint(PyBUF_READ or PyBUF_STRIDED_RO)
  getBuffer(arr, buf, mode)
  result.buf = buf
  result.owner = arr

  # Extract shape
  if result.buf.ndim == 0:
    result.shape = @[result.buf.len div sizeof(T)]
  else:
    result.shape = newSeq[int](result.buf.ndim)
    let shapePtr = cast[ptr UncheckedArray[clong]](result.buf.shape)
    for i in 0..<result.buf.ndim:
      result.shape[i] = int(shapePtr[i])

  # Extract strides
  if result.buf.ndim > 0:
    result.strides = newSeq[int](result.buf.ndim)
    let stridesPtr = cast[ptr UncheckedArray[clong]](result.buf.strides)
    for i in 0..<result.buf.ndim:
      result.strides[i] = int(stridesPtr[i])
  result

proc len*[T](arr: NumpyArrayRead[T]): int {.inline.} =
  result = 1
  for dim in arr.shape:
    result *= dim

proc len*[T](arr: NumpyArrayWrite[T]): int {.inline.} =
  result = 1
  for dim in arr.shape:
    result *= dim

proc shape*[T](arr: NumpyArrayRead[T]): seq[int] {.inline.} =
  arr.shape

proc shape*[T](arr: NumpyArrayWrite[T]): seq[int] {.inline.} =
  arr.shape

proc data*[T](arr: NumpyArrayRead[T]): ptr UncheckedArray[T] {.inline.} =
  cast[ptr UncheckedArray[T]](arr.buf.buf)

proc data*[T](arr: NumpyArrayWrite[T]): ptr UncheckedArray[T] {.inline.} =
  cast[ptr UncheckedArray[T]](arr.buf.buf)

proc `[]`*[T](arr: NumpyArrayRead[T], indices: varargs[int]): T {.inline.} =
  if arr.strides.len == 0:
    # 1D contiguous
    let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
    return data[indices[0]]
  else:
    # Multi-dimensional with strides
    var offset = 0
    for i in 0..<arr.shape.len:
      let strideElements = arr.strides[i] div sizeof(T)
      offset += indices[i] * strideElements

    let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
    return data[offset]

proc `[]`*[T](arr: NumpyArrayWrite[T], indices: varargs[int]): T {.inline.} =
  if arr.strides.len == 0:
    # 1D contiguous
    let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
    return data[indices[0]]
  else:
    # Multi-dimensional with strides
    var offset = 0
    for i in 0..<arr.shape.len:
      let strideElements = arr.strides[i] div sizeof(T)
      offset += indices[i] * strideElements

    let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
    return data[offset]

proc `[]=`*[T](arr: NumpyArrayWrite[T], indices: varargs[int], val: T) {.inline.} =
  if arr.strides.len == 0:
    # 1D contiguous
    let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
    data[indices[0]] = val
  else:
    # Multi-dimensional with strides
    var offset = 0
    for i in 0..<arr.shape.len:
      let strideElements = arr.strides[i] div sizeof(T)
      offset += indices[i] * strideElements

    let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
    data[offset] = val

iterator items*[T](arr: NumpyArrayRead[T]): T =
  let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
  let total = arr.len
  for i in 0..<total:
    yield data[i]

iterator items*[T](arr: NumpyArrayWrite[T]): T =
  let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
  let total = arr.len
  for i in 0..<total:
    yield data[i]

iterator mitems*[T](arr: NumpyArrayWrite[T]): var T =
  let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
  let total = arr.len
  for i in 0..<total:
    yield data[i]

iterator pairs*[T](arr: NumpyArrayRead[T]): tuple[idx: int, val: T] =
  let data = cast[ptr UncheckedArray[T]](arr.buf.buf)
  let total = arr.len
  for i in 0..<total:
    yield (i, data[i])

proc ndim*[T](arr: NumpyArrayRead[T]): int {.inline.} =
  arr.shape.len

proc ndim*[T](arr: NumpyArrayWrite[T]): int {.inline.} =
  arr.shape.len

proc size*[T](arr: NumpyArrayRead[T]): int {.inline.} =
  arr.len

proc size*[T](arr: NumpyArrayWrite[T]): int {.inline.} =
  arr.len

proc isContiguous*[T](arr: NumpyArrayRead[T]): bool {.inline.} =
  arr.strides.len == 0

proc isContiguous*[T](arr: NumpyArrayWrite[T]): bool {.inline.} =
  arr.strides.len == 0
