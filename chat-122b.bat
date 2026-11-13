@echo off
rem Interactive chat with Qwen3.5-122B-A10B (FP8). Usage: chat-122b.bat [extra qwen.exe options]
rem In the chat: /think shows/hides the reasoning, /reset clears the conversation, /exit quits.
rem Startup pins ~108 GB of experts in RAM (about 15-40 s).
cd /d "%~dp0"
title Qwen3.5-122B-A10B (vulkan)
build\Release\qwen.exe --model models\Qwen3.5-122B-A10B-FP8 --backend vulkan --hide-think %1 %2 %3 %4 %5 %6 %7 %8 %9
pause
