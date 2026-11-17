@echo off
rem Runs Noiser headless and waits for it, so scripts get the exit code (noiser.exe is a GUI program, which the shell does not wait for).
rem Usage: noiser-batch.cmd graph.noisegraph --batch-out DIR --batch-dir SRC [options]
start "" /wait "%~dp0noiser.exe" --headless %*
exit /b %errorlevel%
