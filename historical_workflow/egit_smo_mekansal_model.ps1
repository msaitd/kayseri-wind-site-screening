param(
    [Parameter(Mandatory = $true)]
    [string]$WekaJar,

    [string]$JavaExe = "java",

    [string]$Memory = "6g",

    [string]$TrainArff = "",

    [string]$TestArff = "",

    [string]$ModelOut = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($TrainArff)) {
    $TrainArff = Join-Path $WorkDir "weka_veri_hazirligi\mekansal_bagimsiz_test\weka_11f_mekansal_egitim.arff"
}
if ([string]::IsNullOrWhiteSpace($TestArff)) {
    $TestArff = Join-Path $WorkDir "weka_veri_hazirligi\mekansal_bagimsiz_test\weka_11f_mekansal_test.arff"
}
if ([string]::IsNullOrWhiteSpace($ModelOut)) {
    $ModelOut = Join-Path $ScriptDir "smo_mekansal.model"
}

$EvalOut = Join-Path $ScriptDir "smo_mekansal_model_degerlendirme.txt"

if (!(Test-Path -LiteralPath $WekaJar)) { throw "Weka jar bulunamadi: $WekaJar" }
if (!(Test-Path -LiteralPath $TrainArff)) { throw "Egitim ARFF bulunamadi: $TrainArff" }
if (!(Test-Path -LiteralPath $TestArff)) { throw "Test ARFF bulunamadi: $TestArff" }

$args = @(
    "-Xmx$Memory",
    "-cp", $WekaJar,
    "weka.classifiers.functions.SMO",
    "-C", "10.0",
    "-L", "0.001",
    "-P", "1.0E-12",
    "-N", "0",
    "-V", "-1",
    "-W", "1",
    "-K", "weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01",
    "-calibrator", "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4",
    "-t", $TrainArff,
    "-T", $TestArff,
    "-d", $ModelOut
)

Write-Host "SMO modeli egitiliyor..."
Write-Host "Model cikisi:" $ModelOut

& $JavaExe @args *> $EvalOut

if (!(Test-Path -LiteralPath $ModelOut)) {
    throw "Model dosyasi olusmadi. Detay icin bak: $EvalOut"
}

Write-Host "Tamamlandi."
Write-Host "Model:" $ModelOut
Write-Host "Degerlendirme:" $EvalOut
