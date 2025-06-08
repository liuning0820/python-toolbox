# python-toolbox

A collection of small, useful tools written in Python.

## Features

- Various handy scripts for networking, IP calculation, and more
- Easy to extend with your own Python utilities
- Dependency management with [uv](https://github.com/astral-sh/uv) for fast and reproducible installs

## Getting Started

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (install via `pip install uv`)

### Generating requirements.txt

You can generate or update `requirements.txt` with your current environment's dependencies using:

```bash
uv pip freeze > requirements.txt
```

### Install dependencies

If you are in China, you can use the Tsinghua mirror for faster access:

```bash
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Example Tools

- `cidr_calculator.py`: Calculate CIDR notation, network/broadcast address, and IP range for a given IP and subnet mask.
- `get_ip.py`: Fetch your public IP address using an online service.

### Usage

Run any tool directly with Python, for example:

```bash
python cidr_calculator.py
```

or

```bash
python get_ip.py
```

## Adding New Tools

1. Write your Python script in this directory.
2. Add any new dependencies to `requirements.txt`.
3. Install them with:

   ```bash
   uv pip install -r requirements.txt
   ```

