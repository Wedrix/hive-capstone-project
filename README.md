# Hive AI - Hive Weather Inference API

A production-ready FastAPI service for machine learning model inference with support for single and batch predictions. This project is designed for weather classification using a single trained model.

## Features

- 🚀 FastAPI with async support
- 🤖 Weather model loading and inference for weather classification
- 📊 Single and batch prediction endpoints
- 🎨 Modern web UI with Ant Design
- 🏥 Health check and readiness endpoints
- 🐳 Docker support
- 📝 Auto-generated API documentation
- 🔒 CORS configuration
- ⚙️ Environment-based configuration
- 🧪 Comprehensive test suite with pytest
- 🔍 Pre-commit hooks for code quality

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── routes/              # API routes
│   │   ├── __init__.py
│   │   ├── health.py        # Health check endpoints
│   │   └── inference.py     # Weather inference endpoints
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   └── model_service.py # Model loading and prediction
│   └── static/              # Static files
│       └── index.html       # Web UI (Ant Design)
├── models/                  # Weather model storage
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_main.py        # Main app tests
│   ├── test_health.py      # Health endpoint tests
│   ├── test_inference.py   # Inference endpoint tests
│   └── test_model_service.py # Model service tests
├── requirements.txt         # Python dependencies
├── pytest.ini              # Pytest configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── TEAM_MODELS.md          # Team member models documentation
└── README.md              # This file
```

## Setup

### Prerequisites

- Python 3.11+
- pip
- (Optional) Docker and Docker Compose

### Installation

1. **Clone the repository** (if applicable)

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your weather model** (required)
   - Place your trained model file as `models/model.joblib`
   - Place your scaler file as `models/scaler.joblib`
   - The API will return an error if model files are not found

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or using Python:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

## API Endpoints

### Health Checks

- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check (verifies models are loaded)

### Inference

- `POST /api/v1/predict` - Single prediction
  ```json
  {
    "features": [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"]
  }
  ```
  Response:
  ```json
  {
    "prediction": "sunny",
    "confidence": 0.95
  }
  ```

- `POST /api/v1/predict/batch` - Batch predictions
  ```json
  {
    "features": [
      [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"],
      [15.0, 80.0, 25.0, 50.0, 1005.0, 2.0, 5.0, "overcast", "spring", "mountain"]
    ]
  }
  ```

- `GET /api/v1/models` - Get model status

### Web Interface & API Documentation

Once the server is running, visit:
- **Web UI**: http://localhost:8000/home - Interactive Ant Design interface for testing model inferences
  - Single and batch prediction support
  - Real-time API status indicator
  - Weather classification results with visual indicators
- **Root**: http://localhost:8000 - Redirects to Web UI
- **Swagger UI**: http://localhost:8000/docs - Interactive API documentation
- **ReDoc**: http://localhost:8000/redoc - Alternative API documentation

## Configuration

The application uses hardcoded model paths:
- **Model**: `models/model.joblib`
- **Scaler**: `models/scaler.joblib`

No environment configuration is required - simply place your artifacts in the `models/` directory.

## Docker Deployment

### Using Docker Compose

```bash
docker-compose up --build
```

### Using Docker directly

```bash
# Build the image
docker build -t ml-inference-api .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-inference-api
```

## Model Details

This project uses a single weather classification model. The model accepts 10 features:
- 7 numeric features: Temperature, Humidity, Wind Speed, Precipitation, Atmospheric Pressure, UV Index, Visibility
- 3 categorical features: Cloud Cover, Season, Location

The categorical features are transformed into one-hot encoded vectors on the backend before prediction.

### Valid Categorical Values

- **Cloud Cover**: `cloudy`, `clear`, `partly cloudy`, `overcast`
- **Season**: `spring`, `summer`, `autumn`, `fall`, `winter`
- **Location**: `inland`, `mountain`, `coastal`

### Adding Your Own Model

1. **Train and save your model and scaler** using scikit-learn and joblib:
   ```python
   import joblib
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.preprocessing import StandardScaler

   # Train your model
   model = RandomForestClassifier()
   model.fit(X_train, y_train)

   # Save the model
   joblib.dump(model, 'models/model.joblib')

   # Save the scaler
   joblib.dump(scaler, 'models/scaler.joblib')
   ```

2. **Restart the service** - models are loaded on startup

## Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload
```

### Code Quality

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run -a
```

Pre-commit checks include:
- Code formatting (Black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- File validation (YAML, JSON, etc.)

### Code Structure

- `app/main.py` - FastAPI application setup with static file serving
- `app/routes/` - API endpoint definitions
- `app/services/model_service.py` - Model loading and prediction logic with feature transformation
- `app/config.py` - Configuration management
- `app/static/index.html` - Web UI with Ant Design components


## Testing

The project includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_inference.py

# Run tests with verbose output
pytest -v

# Run tests and show coverage in terminal
pytest --cov=app --cov-report=term-missing
```

### Test Coverage

After running tests with coverage, view the HTML report:
```bash
# Generate coverage report
pytest --cov=app --cov-report=html

# Open the report (macOS)
open htmlcov/index.html
```

### Test Structure

- `tests/test_main.py` - Tests for main application and root endpoints
- `tests/test_health.py` - Tests for health check endpoints
- `tests/test_inference.py` - Tests for weather inference endpoints
- `tests/test_model_service.py` - Tests for model service logic
- `tests/conftest.py` - Shared pytest fixtures

### API Testing with curl

Example API calls:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"]
  }'

# Batch prediction
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[25.0, 60.0, 10.0, 0.0, 1013.25, 5.0, 10.0, "cloudy", "summer", "inland"], [15.0, 80.0, 25.0, 50.0, 1005.0, 2.0, 5.0, "overcast", "spring", "mountain"]]
  }'
```

## Weather Classification

This project is designed for weather classification. The model accepts weather features:
- Temperature (°C)
- Humidity (%)
- Wind Speed (km/h)
- Precipitation (%)
- Atmospheric Pressure (hPa)
- UV Index
- Visibility (km)
- Cloud Cover (categorical: cloudy, clear, partly cloudy, overcast)
- Season (categorical: spring, summer, autumn, fall, winter)
- Location (categorical: inland, mountain, coastal)

And predicts weather types such as:
- ☀️ Sunny
- 🌧️ Rainy
- ☁️ Cloudy
- ❄️ Snowy
- ⛈️ Stormy
- 🌫️ Foggy

## Team

This project is developed by the Hive AI team.

## License

MIT
