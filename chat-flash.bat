@echo off
rem Interactive chat with Qwen3.8-Flash-Next (FP8, 180B MoE). Usage: chat-flash.bat [cpu|vulkan] [extra qwen.exe options]
rem In the chat: /think shows/hides the reasoning, /reset clears the conversation, /exit quits.
rem Reasoning effort: add --effort xhigh|medium|low (default xhigh).
cd /d "%~dp0"
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=cpu
title Qwen3.8-Flash-Next (%BACKEND%)
build\Release\qwen.exe --model models\Qwen3.8-Flash-Next-FP8 --backend %BACKEND% --ctx 32768 --hide-think %2 %3 %4 %5 %6 %7 %8 %9
pause
