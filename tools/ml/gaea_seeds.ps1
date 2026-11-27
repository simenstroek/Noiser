# Builds Gaea projects with other mutation seeds through Gaea's Build Swarm, for more training terrains per type.
#   powershell -File tools/ml/gaea_seeds.ps1 [-gaea "E:\Gaea 2.0"] [-dir E:\_NoiserML\gaea] [-seeds 2101..2110]
# Each build goes to <dir>\<Project><seed>\Export.exr (with a copy of the project), named so prep_gaea.py groups it with
# its project; then run prep_gaea.py. Swarm needs its own console window (redirecting its output makes it crash).
param([string]$gaea = "E:\Gaea 2.0", [string]$dir = "E:\_NoiserML\gaea", [int[]]$seeds = (2101..2110),
      [string[]]$projects = @("Alpine", "Craters", "Fantasy01", "Fantasy02", "icelandic", "rockyOutcrops", "sharpRocky", "snowCoveredCliffs"))
foreach ($project in $projects) {
    foreach ($seed in $seeds) {
        $out = Join-Path $dir "$project$seed"
        if (Test-Path (Join-Path $out "Export.exr")) { continue }
        New-Item -ItemType Directory -Force $out | Out-Null
        $t = Get-Date
        $p = Start-Process -FilePath (Join-Path $gaea "Gaea.Swarm.exe") -WorkingDirectory $gaea -PassThru -WindowStyle Minimized `
            -ArgumentList @("--Filename", "`"$(Join-Path $dir "$project.terrain")`"", "--buildpath", "`"$out`"", "--resolution", "4096", "--seed", "$seed", "--silent")
        if (-not $p.WaitForExit(1800000)) { Stop-Process -Id $p.Id -Force; "$project $seed timed out"; continue }
        $ok = Test-Path (Join-Path $out "Export.exr")
        "$project $seed exit $($p.ExitCode) $(if ($ok) { 'ok' } else { 'NO EXPORT' }) $([int]((Get-Date) - $t).TotalSeconds) s"
    }
}
