@echo off
echo [94mSetting up Nepali Sentiment Analysis Project...[0m

:: Create virtual environment
echo [94mCreating virtual environment...[0m
python -m venv venv

:: Activate virtual environment
echo [94mActivating virtual environment...[0m
call venv\Scripts\activate.bat

:: Install requirements
echo [94mInstalling requirements...[0m
pip install -r requirements.txt

:: Check environment setup
echo [94mChecking environment setup...[0m
python check_setup.py

echo [92mSetup completed![0m
echo.
echo To start using the project:
echo [94m1. Train the model:[0m python -m jupyter notebook model_training.ipynb
echo [94m2. Test predictions:[0m python predict.py "तपाईंको फिल्म राम्रो छ"
echo [94m3. Start API server:[0m uvicorn app:app --reload

:: Keep window open
pause