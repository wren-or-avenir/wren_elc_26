# AGENTS.md

## Setup & Dependencies

```bash
pip install opencv-python numpy pyserial
```

Dependencies are **not** declared in any requirements file. The code also uses stdlib `struct`, `queue`, `threading`, `time`, `math`, `enum`.

## Run

```bash
python3 main.py
```

Press `q` to exit.

## Architecture

- `main.py` — entrypoint, control loop, OpenCV trackbar GUI, UART sender
- `models/` — implicit namespace package (no `__init__.py`, works in Python 3)
  - `cam.py` — threaded camera capture, MJPG 640×480, attempts indices 4..24
  - `detector.py` — ball detection via adaptive threshold + 1D Y-projection
  - `Kalman.py` — 1D Kalman filter (position+velocity state)
  - `tracker.py` — tracking state machine (LOST/TMP_LOST/TRACK), wraps Kalman
  - `uart.py` — binary UART protocol at 115200 baud

## Hardware

- Requires a camera at index 4 and a serial device at `/dev/ttyACM0`
- UART frame: `0x55 0xAA | y_offset(f32 LE) | y_vel(f32 LE) | status(u8) | checksum(u8) | 0x0A`

## Branches

| Branch | Detection approach |
|---|---|
| `study_type` (current) | Linear 1D Y-projection |
| `main` | Prior detector algorithm |
| `hough_circle` | Hough circle transform |

## Tooling

No tests, linter, formatter, type checker, or CI are configured.

## ruling
#"写代码的时前遵循四点：1.如果不确定，请询问而非猜测；当存在歧义时，不要默默选择；如果有更简单的方法，就直言不讳；指出不清楚的地方并寻求澄清。"
#"2.除了被要求的内容之外，没有其他功能；一次性代码无抽象；没有“灵活性”或“可配置性”，只要是没被要求的；对于不可能的情景，没有错误处理；如果200行可以变成50行，那就重写它。""
#""3.不要“改进”相邻的代码、注释或格式；不要重构那些没坏掉的东西；即使你会用不同的方式，也要匹配现有的风格；如果你发现了无关的死代码，要提及——不要删除。"
#""4.对于多步骤任务，请提出简要计划。""