# PCB Microscope Scanner

Сканер печатных плат

## Требования

- Python 3.10+
- GRBL-совместимый стол (Creality Falcon2 и т.п.) на COM-порту
- UVC-камера (микроскоп)

## Установка

```Windows powershell
python -m pip install fastapi "uvicorn[standard]" opencv-python pyserial numpy python-multipart
```
```Linux
sudo apt update
sudo apt install -y python3 python3-pip python3-venv \
    libopencv-dev python3-opencv v4l-utils
sudo usermod -aG dialout $USER
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" opencv-python pyserial numpy python-multipart
```

## Запуск

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level warning
```
```Linux
uvicorn app:app --host 0.0.0.0 --port 8000
```

Затем открыть http://localhost:8000 в браузере

## Особенности использования

1. При подключении в поле статуса должно появиться что-то вроде Idle X:0.000 Y:0.000,
если появляется Alarm - нажми XUnlock, потом HHome. Каретка уедет в нулевую точку.
2. Эта команда позволит увидеть индекс камеры, по стандарту - 0
```powershell
python -c "import cv2; print([ (i, cv2.VideoCapture(i, cv2.CAP_MSMF).isOpened()) for i in range(5) ])"
```
3. Параметры перемещения: шаг в мм, скорость перемещения в мм/мин
4. Настройки сканирования: количество колонок и рядов, шаг по осям в мм, settle - пауза после остановки в секундах, 
bool Snake Pattern - перемещение змейкой, где четные ряды идут справа налево.