# Starts Noiser with the given arguments, waits, and saves a screenshot of its window (the desktop must be unlocked).
# Example: powershell -File tools/screenshot.ps1 -out shot.png -appArgs @('--mesh-graph','E:\g.meshgraph','--run-mesh-graph','--window','1500x900') -wait 30
# A path with spaces needs its own quotes: '"H:\My Drive.obj"'.
param([string]$out, [string[]]$appArgs, [int]$wait = 9)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
}
"@
$p = Start-Process -FilePath "E:\_Noiser\target\release\noiser.exe" -ArgumentList $appArgs -PassThru
Start-Sleep -Seconds $wait
$p.Refresh()
if ($p.HasExited) { "exited: $($p.ExitCode)"; exit }
[W]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 800
$r = New-Object W+RECT
[W]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
$w = $r.R - $r.L; $h = $r.B - $r.T
"window $w x $h at $($r.L),$($r.T)"
$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($r.L, $r.T, 0, 0, (New-Object System.Drawing.Size $w, $h))
$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
Stop-Process -Id $p.Id -Force
"saved $out"
