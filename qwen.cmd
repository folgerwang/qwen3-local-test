@echo off
rem qwen [35b|122b|flash] [options]  -  chat with a local Qwen model from any directory.
rem   qwen                 Qwen3.6-35B-A3B (default)
rem   qwen 122b -c         Qwen3.5-122B-A10B, continuing the last conversation
rem   qwen flash           Qwen3.8-Flash-Next (CPU backend unless --backend vulkan)
rem   qwen 35b -p "hi"     one-shot prompt; qwen --help lists every option
rem Add this folder to PATH to use it anywhere. In the chat, /help lists the commands.
setlocal
set "ROOT=%~dp0"
set "MODEL=Qwen3.6-35B-A3B-FP8"
set "BACKEND=vulkan"
if /i "%~1"=="35b"   (set "MODEL=Qwen3.6-35B-A3B-FP8"    & shift)
if /i "%~1"=="122b"  (set "MODEL=Qwen3.5-122B-A10B-FP8"  & shift)
if /i "%~1"=="flash" (set "MODEL=Qwen3.8-Flash-Next-FP8" & set "BACKEND=cpu" & shift)
set "ARGS="
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
if not exist "%ROOT%models\%MODEL%\config.json" (
    echo Model %MODEL% is not downloaded. Run: setup.bat --model ^<35b^|122b^|flash^>
    exit /b 1
)
title qwen - %MODEL% - %CD%
"%ROOT%build\Release\qwen.exe" --model "%ROOT%models\%MODEL%" --backend %BACKEND% --ctx 32768 --hide-think%ARGS%
