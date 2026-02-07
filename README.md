# Dispatcher-Truck Multi-Agent Reinforcement Learning System

    A hierarchical MARL system for optimizing logistics dispatch and routing with real-time traffic and weather integration.

    ## Features

    - 🚛 Hierarchical multi-agent system (Dispatcher + Trucks)
    - 🎮 Gymnasium-compliant environment
    - 🧠 PPO/MAPPO training with curriculum learning
    - 🌐 FastAPI backend for inference
    - ⚛️ React + Vite frontend (coming soon)
    - 🌦️ Real-time traffic and weather integration
    - 🚨 Comprehensive edge case handling
    - ☁️ AWS deployment ready

    ## Quick Start

    ### Installation
```bash
    # Clone or download the project
    cd dispatcher-truck-marl

    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate

    # Install dependencies
    pip install -e .
```

    ### Test Installation
```bash
    python test_installation.py
```

    ### Train
```bash
    # Basic training
    python training/train.py --timesteps 100000

    # With custom parameters
    python training/train.py --timesteps 500000 --trucks 10 --routes 50
```

    ### Run API Server
```bash
    python marl_api/main.py
    # or
    uvicorn marl_api.main:app --reload
```

    ## Project Structure
```
    dispatcher-truck-marl/
    ├── marl_env/           # Gymnasium environment
    ├── policies/           # RL policy networks
    ├── training/           # Training scripts
    ├── external_apis/      # Traffic/weather APIs
    ├── edge_cases/         # Edge case handlers
    ├── marl_api/           # FastAPI backend
    ├── tests/              # Unit tests
    └── configs/            # Configuration files
```

    ## Training

    The system uses PPO (Proximal Policy Optimization) for training both the dispatcher and truck policies.

    ### Basic Training
```bash
    python training/train.py
```

    ### Resume from Checkpoint
```bash
    python training/train.py --checkpoint checkpoints/ppo_dispatcher_truck
```

    ### Evaluation
```bash
    python training/evaluate.py --model checkpoints/ppo_dispatcher_truck --episodes 20
```

    ## API Usage

    ### Create Environment
```bash
    curl -X POST http://localhost:8000/api/v1/environments \
      -H "Content-Type: application/json" \
      -d '{"config": {"num_trucks": 5, "num_routes": 20}}'
```

    ### Step Environment
```bash
    curl -X POST http://localhost:8000/api/v1/predict \
      -H "Content-Type: application/json" \
      -d '{"environment_id": "env_0", "num_steps": 1}'
```

    ### Get State
```bash
    curl http://localhost:8000/api/v1/environments/env_0/state
```

    ## Configuration

    Edit `configs/training_config.yaml` for training parameters.
    Edit `configs/deployment_config.yaml` for API settings.

    ## Testing
```bash
    # Run all tests
    pytest tests/

    # Run specific test
    pytest tests/test_environment.py -v

    # With coverage
    pytest tests/ --cov=. --cov-report=html
```

    ## External API Integration

    To use real-time traffic and weather data:

    1. Get API keys:
       - TomTom Traffic: https://developer.tomtom.com/
       - OpenWeatherMap: https://openweathermap.org/api

    2. Add to `.env`:
```
       TOMTOM_API_KEY=your_key_here
       OPENWEATHER_API_KEY=your_key_here
```

    ## Troubleshooting

    ### Import Errors

    If you get import errors, make sure you've installed the package:
```bash
    pip install -e .
```

    ### Gymnasium Environment Errors

    Check the environment is valid:
```python
    from gymnasium.utils.env_checker import check_env
    from marl_env import DispatcherTruckEnv

    env = DispatcherTruckEnv(num_trucks=3)
    check_env(env)
```

    ## Contributing

    Contributions are welcome! Please:
    1. Fork the repository
    2. Create a feature branch
    3. Make your changes
    4. Add tests
    5. Submit a pull request

    ## License

    MIT License - see LICENSE file for details

    ## Citation

    If you use this code in your research, please cite:
```bibtex
    @software{dispatcher_truck_marl,
      title = {Dispatcher-Truck MARL System},
      author = {Your Name},
      year = {2024},
      url = {https://github.com/yourusername/dispatcher-truck-marl}
    }
```

    ## Acknowledgments

    - Built with Gymnasium, PyTorch, and Stable-Baselines3
    - Traffic data from TomTom API
    - Weather data from OpenWeatherMap API
