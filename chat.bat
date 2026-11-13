@echo off
rem Interactive chat with Qwen3.8-Flash-Next 128B (UD-Q4_K_XL 4-bit GGUF). Usage: chat.bat [cpu|vulkan] [extra qwen.exe options]
rem In the chat: /think shows/hides the reasoning, /reset clears the conversation, /exit quits.
rem The vulkan backend keeps the dense weights in VRAM and fills the VRAM that is free with a cache of hot experts (--expert-cache MB caps it, 0 = none);
rem the other experts are computed by the CPU from the ~72 GB of expert weights in RAM.
cd /d "%~dp0"
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=vulkan
title Qwen3.8-Flash-Next UD-Q4_K_XL (%BACKEND%)
build\Release\qwen.exe --backend %BACKEND% --hide-think %2 %3 %4 %5 %6 %7 %8 %9
pause
