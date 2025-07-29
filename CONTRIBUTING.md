# Contributing to Climate Policy Impact Analyzer

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nimmmalarohit/climate_risk_scenario_generation.git
   cd climate_risk_scenario_generation
   ```

2. **Set up development environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure API key**
   ```bash
   mkdir -p secrets
   echo "your-actual-openai-api-key" > secrets/OPENAI_API_KEY.txt
   ```

## Running in Development Mode

```bash
export FLASK_DEBUG=true
export PORT=5000
python3 start_ui.py
```

## Code Style

- Follow PEP 8 formatting
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions under 50 lines when possible

## Testing

### Automated Testing

Run the integration test suite:
```bash
python3 tests/integration_test.py
```

This tests all API endpoints, error handling, and rate limiting.

### Manual Testing

1. Start the UI: `python3 start_ui.py`
2. Test query: "What if California implements carbon pricing at $75/ton by 2027?"
3. Verify all tabs load correctly
4. Check that charts are generated

## Pull Request Process

1. Create a feature branch from main
2. Make your changes
3. Run automated tests: `python3 tests/integration_test.py`
4. Test the application manually
5. Update documentation if needed
6. Submit pull request with clear description

## Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: False)
- `OPENAI_API_KEY`: Can be set as env var instead of file

## Architecture

- `src/climate_risk_scenario_generation/core/`: Core analysis logic
- `src/climate_risk_scenario_generation/data/`: Data providers
- `src/climate_risk_scenario_generation/models/`: Economic models
- `src/climate_risk_scenario_generation/visualization/`: Chart generation
- `templates/`: HTML templates
- `static/`: Static assets and generated charts