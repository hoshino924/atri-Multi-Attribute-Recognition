param(
    [string]$SourceDirectory = (Join-Path $PSScriptRoot '../icons'),
    [string]$OutputPath = (Join-Path $PSScriptRoot '../icons/atri.ico')
)

# PNG payloads are copied verbatim. Never derive the smaller artwork from 256px.
# ICO layout: https://devblogs.microsoft.com/oldnewthing/20101018-00/?p=12513
# PNG payloads: https://devblogs.microsoft.com/oldnewthing/20101022-00/?p=12473
$ErrorActionPreference = 'Stop'
$sizes = @(16, 24, 32, 48, 64, 128, 256)
$frames = foreach ($size in $sizes) {
    $path = Join-Path $SourceDirectory "icon-$size.png"
    $bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $path).Path)
    if ($bytes.Length -lt 33 -or
        [BitConverter]::ToString($bytes, 0, 8) -ne '89-50-4E-47-0D-0A-1A-0A' -or
        [Text.Encoding]::ASCII.GetString($bytes, 12, 4) -ne 'IHDR') {
        throw "Not a PNG with an IHDR header: $path"
    }
    $width = [int]$bytes[16] * 16777216 + [int]$bytes[17] * 65536 + [int]$bytes[18] * 256 + $bytes[19]
    $height = [int]$bytes[20] * 16777216 + [int]$bytes[21] * 65536 + [int]$bytes[22] * 256 + $bytes[23]
    if ($width -ne $size -or $height -ne $size -or $bytes[24] -ne 8 -or $bytes[25] -notin @(2, 6)) {
        throw "Expected $size x $size, 8-bit RGB or RGBA PNG: $path"
    }
    [pscustomobject]@{ Size = $size; Data = $bytes; Bits = $(if ($bytes[25] -eq 6) { 32 } else { 24 }) }
}

$target = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $target) { throw "Output already exists: $target" }
$stream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
$writer = [IO.BinaryWriter]::new($stream)
try {
    $writer.Write([uint16]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]$frames.Count)
    $offset = 6 + 16 * $frames.Count
    foreach ($frame in $frames) {
        $dimension = if ($frame.Size -eq 256) { 0 } else { $frame.Size }
        $writer.Write([byte]$dimension)
        $writer.Write([byte]$dimension)
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]$frame.Bits)
        $writer.Write([uint32]$frame.Data.Length)
        $writer.Write([uint32]$offset)
        $offset += $frame.Data.Length
    }
    foreach ($frame in $frames) { $writer.Write([byte[]]$frame.Data) }
} finally {
    $writer.Dispose()
}

# Read back the directory and every embedded PNG, including the small variants.
$result = [IO.File]::ReadAllBytes($target)
if ([BitConverter]::ToUInt16($result, 4) -ne $frames.Count) { throw 'Frame count mismatch' }
$sha = [Security.Cryptography.SHA256]::Create()
try {
    for ($index = 0; $index -lt $frames.Count; $index++) {
        $entry = 6 + $index * 16
        $length = [BitConverter]::ToUInt32($result, $entry + 8)
        $position = [BitConverter]::ToUInt32($result, $entry + 12)
        $frame = $frames[$index]
        $dimension = if ($frame.Size -eq 256) { 0 } else { $frame.Size }
        if ($result[$entry] -ne $dimension -or $result[$entry + 1] -ne $dimension -or $length -ne $frame.Data.Length) {
            throw "ICO entry differs from source: $($frame.Size)"
        }
        $embeddedHash = [BitConverter]::ToString($sha.ComputeHash($result, [int]$position, [int]$length))
        $sourceHash = [BitConverter]::ToString($sha.ComputeHash([byte[]]$frame.Data))
        if ($embeddedHash -ne $sourceHash) { throw "PNG payload changed: $($frame.Size)" }
        Write-Output "$($frame.Size)x$($frame.Size): source PNG preserved exactly"
    }
} finally {
    $sha.Dispose()
}
Write-Output "Created $target"
