@echo off
FOR %%i IN (*.ljp) DO (
    echo %%i
    D:\Build\_vc17-x64\bin\djpeg.exe %%i 1> %%i.pam
    echo.
)