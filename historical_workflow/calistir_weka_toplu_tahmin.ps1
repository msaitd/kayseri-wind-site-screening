param(
    [Parameter(Mandatory = $true)]
    [string]$WekaJar,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$JavaExe = "java",

    [int]$Parallel = 1,

    [string]$Memory = "4g",

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ChunkDir = Join-Path $ScriptDir "arff_chunks"
$OutDir = Join-Path $ScriptDir "weka_tahmin_sonuclari"
$LogDir = Join-Path $OutDir "logs"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (!(Test-Path -LiteralPath $WekaJar)) {
    throw "Weka jar bulunamadi: $WekaJar"
}
if (!(Test-Path -LiteralPath $Model)) {
    throw "Weka model dosyasi bulunamadi: $Model"
}
if (!(Test-Path -LiteralPath $ChunkDir)) {
    throw "ARFF chunk klasoru bulunamadi: $ChunkDir"
}

$chunks = Get-ChildItem -LiteralPath $ChunkDir -Filter "*.arff" | Sort-Object Name
if ($chunks.Count -eq 0) {
    throw "ARFF chunk bulunamadi: $ChunkDir"
}

if ($Parallel -lt 1) {
    $Parallel = 1
}

Write-Host "Toplam chunk:" $chunks.Count
Write-Host "Paralel islem:" $Parallel
Write-Host "Cikti klasoru:" $OutDir

$running = New-Object System.Collections.Generic.List[object]

function Quote-ProcessArgument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }

    return $Value
}

function Wait-OneFinished {
    param([System.Collections.Generic.List[object]]$Running)
    while ($true) {
        for ($i = $Running.Count - 1; $i -ge 0; $i--) {
            $item = $Running[$i]
            $item.Process.Refresh()
            if ($item.Process.HasExited) {
                $exitCode = $item.Process.ExitCode
                $stderrLength = 0
                $outLength = 0
                if (Test-Path -LiteralPath $item.ErrPath) {
                    $stderrLength = (Get-Item -LiteralPath $item.ErrPath).Length
                }
                if (Test-Path -LiteralPath $item.OutPath) {
                    $outLength = (Get-Item -LiteralPath $item.OutPath).Length
                }
                if (($null -ne $exitCode) -and ($exitCode -ne 0)) {
                    throw "Weka hata verdi: $($item.ChunkName), ExitCode=$($item.Process.ExitCode). Log: $($item.ErrPath)"
                }
                if (($null -eq $exitCode) -and (($stderrLength -gt 0) -or ($outLength -eq 0))) {
                    throw "Weka sonucu dogrulanamadi: $($item.ChunkName), ExitCode bos. Log: $($item.ErrPath)"
                }
                Write-Host "[ok]" $item.ChunkName "->" $item.OutPath
                $Running.RemoveAt($i)
                return
            }
        }
        Start-Sleep -Seconds 2
    }
}

foreach ($chunk in $chunks) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($chunk.Name)
    $outPath = Join-Path $OutDir ($base + "_predictions.txt")
    $errPath = Join-Path $LogDir ($base + "_stderr.txt")

    if ((Test-Path -LiteralPath $outPath) -and !$Overwrite) {
        Write-Host "[skip]" $chunk.Name "cikti zaten var"
        continue
    }

    while ($running.Count -ge $Parallel) {
        Wait-OneFinished -Running $running
    }

    $javaArgs = @(
        "-Xmx$Memory",
        "-cp", $WekaJar,
        "weka.classifiers.functions.SMO",
        "-l", $Model,
        "-T", $chunk.FullName,
        "-p", "0"
    )
    $argLine = ($javaArgs | ForEach-Object { Quote-ProcessArgument $_ }) -join " "

    Write-Host "[start]" $chunk.Name
    $proc = Start-Process -FilePath $JavaExe -ArgumentList $argLine -NoNewWindow -PassThru `
        -RedirectStandardOutput $outPath -RedirectStandardError $errPath

    $running.Add([pscustomobject]@{
        Process = $proc
        ChunkName = $chunk.Name
        OutPath = $outPath
        ErrPath = $errPath
    }) | Out-Null
}

while ($running.Count -gt 0) {
    Wait-OneFinished -Running $running
}

Write-Host "Tamamlandi. Tahmin dosyalari:" $OutDir
