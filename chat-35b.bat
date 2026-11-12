@echo off
rem Interactive chat with Qwen3.6-35B-A3B (FP8). Usage: chat-35b.bat [cpu|vulkan] [extra qwen.exe options]
rem In the chat: /think shows/hides the reasoning, /reset clears the conversation, /exit quits.
cd /d "%~dp0"
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=vulkan
title Qwen3.6-35B-A3B (%BACKEND%)
build\Release\qwen.exe --model models\Qwen3.6-35B-A3B-FP8 --backend %BACKEND% --ctx 32768 --hide-think %2 %3 %4 %5 %6 %7 %8 %9
pause
