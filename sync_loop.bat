@echo off
echo 🔄 Starting rclone sync loop with delete...
:loop
"C:\Users\maria selciya\rclone-v1.70.3-windows-amd64\rclone-v1.70.3-windows-amd64\rclone.exe" sync gdrivekb_rclone:chatbot-kb "C:\Users\maria selciya\Desktop\chatbotKB_test" --delete-excluded -v
timeout /t 30 >nul
goto loop




