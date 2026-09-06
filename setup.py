from setuptools import setup, find_packages

setup(
    name="portal_bot",
    version="1.0.0",
    description="Autonomous Bot for Portal 1 via Real-Time Computer Vision and Closed-Loop Control",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "mss",
        "Pillow",
        "fastapi",
        "uvicorn",
        "websockets",
        "pydantic",
        "pyautogui",
    ],
    entry_points={
        "console_scripts": [
            "portal-bot=run_bot:main",
        ],
    },
)
