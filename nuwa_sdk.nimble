# Package

version       = "0.4.2"
author        = "Martin Eastwood"
description   = "SDK for Nuwa Build - provides compile-time metadata for Python stub generation and numpy array wrappers"
license       = "MIT"
srcDir        = "src"

# Dependencies

requires "nim >= 1.6.0"
requires "nimpy >= 0.2.0"

# Tests

task test, "Run all tests":
  exec "nim c -r --path:src tests/test_numpy.nim"