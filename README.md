# Climate Policy Impact Analyzer

A quantitative climate policy risk analyzer that combines economic models with large language models to assess financial impacts of climate policies.

## Quick Start

**Just want to try it out?** Here's the minimal setup:

1. Make sure you have Python 3.8+ and pip installed
2. Clone the project: `git clone https://github.com/nimmmalarohit/climate_risk_scenario_generation.git`
3. Open terminal in the project directory: `cd climate_risk_scenario_generation`
4. Run these commands:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   mkdir -p secrets
   echo "sk-proj-your-actual-openai-key-here" > secrets/OPENAI_API_KEY.txt
   python3 start_ui.py
   ```
   
   On Windows, use `venv\Scripts\activate` instead of `source venv/bin/activate`
5. Open http://localhost:5000 in your browser
6. Try a query like: "What if California implements carbon pricing at $75/ton by 2027?"

**Need help?** See the detailed instructions below.

## Table of Contents

1. [System Overview](#system-overview)
2. [Economic Models](#economic-models)
3. [Setup Instructions](#setup-instructions)
4. [OpenAI and Ollama Configuration](#openai-and-ollama-configuration)
5. [Parameter Tuning](#parameter-tuning)
6. [Launching the UI](#launching-the-ui)
7. [Navigating the Interface](#navigating-the-interface)
8. [Understanding Results](#understanding-results)
9. [Data Sources](#data-sources)
10. [Testing](#testing)
11. [Limitations](#limitations)
12. [Troubleshooting](#troubleshooting)

## System Overview

The Climate Policy Impact Analyzer analyzes "what-if" climate policy scenarios using real quantitative economic models combined with language model interpretation. 

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │    │  Policy Parser   │    │ Economic Models │
│ "What if CA     │───►│ Extracts params  │───►│ Transport, Carbon│
│  bans gas cars  │    │ action, timeline │    │ Renewable, etc. │
│  by 2030?"      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Results UI    │◄───│ Integrated       │◄───│ LLM Interpreter │
│ Charts, metrics │    │ Analysis Engine  │    │ OpenAI/Ollama   │
│ Risk assessment │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **Policy Parser**: Extracts structured parameters from natural language queries
- **Economic Models**: Quantitative models for transport, carbon pricing, renewable energy, and regulation
- **LLM Integration**: Uses OpenAI or local Ollama models for interpretation 
- **NGFS Integration**: Aligns results with Network for Greening the Financial System scenarios
- **Web UI**: Interactive interface for query input and result visualization

## Economic Models

The system uses quantitative economic impact models for climate policies with empirical validation and financial analysis.

### Supported Policy Types

#### Transport Electrification
- EV mandates and targets
- Gasoline vehicle bans
- EV tax credits and incentives

#### Carbon Pricing
- Carbon taxes
- Cap-and-trade systems
- Sectoral carbon pricing

#### Renewable Energy
- Renewable portfolio standards
- Clean energy investment programs
- Renewable energy mandates

#### Fossil Fuel Regulation
- Federal land drilling restrictions
- Production limits and quotas
- Extraction regulations

### Model Features

- Market maturity modeling with S-curve adoption patterns
- Net present value calculations using discount rates
- General equilibrium adjustments for economic spillovers
- Correlation-adjusted uncertainty quantification
- Historical validation against real policy experiences
- Input validation and error bounds checking

### Mathematical Approach

#### Market Maturity (Transport Policies)
```
maturity = 1 / (1 + exp(-k * (years - t0)))
```
Where k=0.4 (growth rate) and t0=8 (50% adoption at 8 years)

#### Financial Discounting
```
pv_factor = 1 / ((1 + discount_rate) ^ years)
```
Using 2% social discount rate for policy impacts, 3% for infrastructure

#### Uncertainty Combination with Correlations
```
σ_total² = u^T * Σ * u
```
Where u is uncertainty vector, Σ is correlation matrix

#### Carbon Pricing Impact
```
cost_increase = (carbon_intensity * carbon_price) / 1000
output_change = cost_increase * price_elasticity
```
Using EPA EEIO carbon intensities and elasticities

### Model Data Sources

- Policy effectiveness: ICCT (2021), IEA Global EV Outlook 2023, Norway EV data
- Economic multipliers: BLS Input-Output tables 2022, DOE Employment Report 2023
- Carbon intensities: EPA EEIO model v2.0 (2022), BEA Industry Accounts
- Price elasticities: Labandeira et al. (2017) meta-analysis, CBO (2022)
- Regional emissions: EPA State GHG Inventories 2023
- Investment requirements: NREL Electrification Futures Study (2023)

### Model Ranges

#### Transport Electrification
- Based on Norway (85% EV share), California ZEV program
- GDP impacts: -0.5% to +0.3%
- Timeline: 5-15 years

#### Carbon Pricing
- Based on BC carbon tax, EU ETS, RGGI experience
- Price range: $10-200/tCO2
- GDP impacts: -2% to +0.5%

#### Renewable Energy
- Based on German Energiewende, Denmark wind expansion
- Target range: 20-100% renewable share
- Timeline: 2025-2050

### Model Usage Example

```python
from .generic_policy_model import GenericPolicyModelFramework
from ..core.policy_parser import PolicyParameterParser

# Initialize framework
framework = GenericPolicyModelFramework()
parser = PolicyParameterParser()

# Parse and analyze policy
params = parser.parse("What if California bans gas cars by 2030?")
impact = framework.calculate_policy_impact(params)

# Access results
gdp_impact = impact.economic_impact['gdp_impact_percent']
sectoral_impacts = impact.sectoral_impacts
uncertainty = impact.uncertainty_bounds
```

### Output Format

All models return a `PolicyImpact` object containing:

- `economic_impact`: GDP percentage change, employment effects, investment shifts
- `sectoral_impacts`: Industry-specific impacts (automotive, electricity, oil/gas, etc.)
- `temporal_effects`: Time-phased impact evolution (immediate, short-term, medium-term, long-term)
- `uncertainty_bounds`: Statistical confidence intervals for predictions
- `model_metadata`: Model type, parameters, and calculation details

## Setup Instructions

### Prerequisites

- **Python 3.8 or higher** - Check with `python3 --version`
- **pip** - Usually comes with Python, check with `pip3 --version`
- **Git** (optional) - Only needed if cloning from repository
- **Internet connection** - Required for OpenAI models and initial setup
- **4GB+ RAM** - Recommended for smooth operation
- **Optional: Ollama** - For free local models (alternative to OpenAI)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nimmmalarohit/climate_risk_scenario_generation.git
   cd climate_risk_scenario_generation
   ```
   
   **Alternative**: Download as ZIP from GitHub:
   - Go to https://github.com/nimmmalarohit/climate_risk_scenario_generation
   - Click "Code" → "Download ZIP"
   - Extract and navigate to the folder

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   On Windows: `venv\Scripts\activate`
   
   After activation, your terminal prompt should show `(venv)` at the beginning.

3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

4. **Install spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```
   
   If this fails, try:
   ```bash
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
   ```

5. **Create secrets directory and API key file**
   ```bash
   mkdir -p secrets
   touch secrets/OPENAI_API_KEY.txt
   ```
   
   **Important**: You must add your OpenAI API key to this file before the system will work.

6. **Verify installation**
   ```bash
   python3 -c "import sys; sys.path.insert(0, 'src'); from climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer; print('Installation verified successfully')"
   ```
   
   If you see "Installation verified successfully", you're ready to proceed.

## OpenAI and Ollama Configuration

### OpenAI Setup

1. **Get API Key**
   - Sign up at https://platform.openai.com/
   - Generate an API key from the dashboard
   - Note: API usage incurs costs based on tokens processed

2. **Configure API Key**
   
   Replace with your actual API key:
   ```bash
   echo "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > secrets/OPENAI_API_KEY.txt
   ```
   
   Or edit the file directly with `nano secrets/OPENAI_API_KEY.txt` (Linux/Mac) or `notepad secrets/OPENAI_API_KEY.txt` (Windows).
   
   The API key should be the only content in this file, no quotes or extra spaces.

3. **Available Models and Costs**
   - GPT-3.5 Turbo: $0.008/query (fastest, most economical)
   - GPT-4o Mini: $0.024/query (good balance)
   - GPT-4o: $0.12/query (most capable)
   - GPT-4 Turbo: $0.18/query (high performance)
   - GPT-4: $0.45/query (original, most expensive)

### Ollama Setup (Optional - Free Local Models)

1. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
   
   On Windows: Download from https://ollama.ai/download

2. **Download Models**
   ```bash
   ollama pull llama3.1:8b
   ollama pull llama3.1:13b
   ```

3. **Start Ollama Server**
   ```bash
   ollama serve
   ```
   
   Server runs on http://localhost:11434

4. **Verify Installation**
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Parameter Tuning

### Model Selection Guidelines

**Use OpenAI when**:
- Highest accuracy is required
- Budget allows for API costs
- Internet connection is reliable

**Use Ollama when**:
- Privacy is important (local processing)
- No API costs desired
- Offline capability needed

### Economic Model Parameters

The system uses built-in economic parameters calibrated from real data:

- **Transport Electrification**: S-curve adoption, market maturity factors
- **Carbon Pricing**: Marginal abatement cost curves, price elasticities  
- **Renewable Energy**: Capacity factors, grid integration costs
- **GDP Impact**: Keynesian multipliers, regional adjustments

These parameters are embedded in the models and don't require user tuning.

## Launching the UI

### Before First Launch

Make sure you have:
1. Activated your virtual environment (`source venv/bin/activate`)
2. Added your OpenAI API key to `secrets/OPENAI_API_KEY.txt`
3. Verified the installation works

### Basic Launch

```bash
python3 start_ui.py
```

Always use `python3` explicitly, as `python` may point to Python 2 on some systems.

### Expected Output

When you run `python3 start_ui.py`, you should see:

```
======================================================================
QUANTITATIVE CLIMATE POLICY RISK ANALYZER - WEB UI
======================================================================

Starting Quantitative Climate Policy Risk Analyzer...
Features:
- Real quantitative economic models for accurate calculations
- NGFS scenario integration for financial risk assessment
- Timeline-sensitive impact modeling (2025 vs 2030 matters\!)
- Validation framework with confidence intervals
- LLM interpretation of quantitative results

The web interface will be available at: http://localhost:5000
Press Ctrl+C to stop the server
```

If you see errors instead, check the [Troubleshooting](#troubleshooting) section.

### Access the Interface

1. **Open your web browser** (Chrome, Firefox, Safari, etc.)
2. **Navigate to**: `http://localhost:5000`
3. **You should see** a blue gradient page with "Climate Policy Impact Analyzer" at the top
4. **If the page doesn't load**, check that:
   - The terminal shows "Running on http://127.0.0.1:5000"
   - No error messages appear in the terminal
   - You're using the correct URL (localhost:5000, not 127.0.0.1:5000)

## Navigating the Interface

### Main Interface Elements

1. **Model Selection Dropdown**
   - Choose between OpenAI and Ollama models
   - Shows cost per query for OpenAI models
   - Shows "FREE" for local Ollama models

2. **Query Input**
   - Large text area for entering policy questions
   - Supports natural language queries
   - Examples provided for guidance

3. **Example Queries**
   - Click any example to populate the query field
   - Covers various policy types and scenarios

### Query Examples

- `What if California implements carbon pricing at $75/ton by 2027?`
- `What happens if the Fed stops EV credits by 2026?`
- `What if Texas bans gas cars by 2030?`
- `What if Europe implements $200/ton carbon tax by 2028?`

### Result Tabs

**Overview Tab**
- Key metrics: Total Effects, Feedback Loops, Shock Magnitude
- Risk assessment with color-coded levels
- Parsed query parameters and confidence scores

**Cascade Effects Tab**  
- First-order effects (0-6 months)
- Second-order effects (6-24 months) 
- Third-order effects (2-5 years)

**Feedback Loops Tab**
- Reinforcing loops (amplify effects)
- Balancing loops (stabilize effects)
- Tipping point dynamics

**Charts & Graphs Tab**
- Risk assessment pie chart
- Sector impact bar chart
- Timeline effects visualization
- Confidence metrics radar chart

**Data Sources Tab**
- Lists all data sources used
- Provides transparency on information basis

## Understanding Results

### Key Metrics Interpretation

**Total Effects (15-50)**
- Number of cascading impacts across all domains
- Higher numbers indicate broader policy impacts

**Shock Magnitude (0-5)**
- Initial policy impact intensity
- Values above 3.0 indicate significant disruption

**Cumulative Impact (0-20)**  
- Total system-wide effect after all cascades
- Higher values suggest major economic transformation

**Feedback Loops**
- Self-reinforcing or self-regulating dynamics
- More loops indicate complex system interactions

### Risk Assessment Levels

**Low Risk**
- GDP impact < 0.5%
- Standard monitoring sufficient
- Green color coding

**Medium Risk**
- GDP impact 0.5-2.0%
- Develop contingency plans
- Yellow color coding

**High Risk**
- GDP impact > 2.0%
- Immediate assessment required  
- Red color coding

### Timeline Understanding

**First-Order (0-6 months)**
- Direct, immediate policy impacts
- Regulatory compliance costs
- Market price adjustments

**Second-Order (6-24 months)**
- Market and actor responses
- Supply chain adjustments
- Investment redirections

**Third-Order (2-5 years)**
- Long-term structural changes
- Technology adoption curves
- Macroeconomic adjustments

## Data Sources

The system integrates **real-time economic data** from multiple authoritative government sources to provide accurate baseline conditions and validate policy impact predictions. All data sources use official APIs with proper authentication.

### Real-Time Economic Data APIs

#### 1. FRED (Federal Reserve Economic Data)
**Provider**: Federal Reserve Bank of St. Louis  
**Website**: https://fred.stlouisfed.org/  
**Data Used**: 
- GDP (Gross Domestic Product)
- Unemployment rates (UNRATE)
- Consumer Price Index (CPIAUCSL)
- Industrial Production (INDPRO)
- Federal Funds Rate (FEDFUNDS)
- Energy prices and consumption
- Sectoral employment data

**API Registration**: https://fred.stlouisfed.org/docs/api/api_key.html  
**Rate Limit**: 120 calls per hour  
**Historical Coverage**: 900+ data points per series (monthly/quarterly data back to 1940s)

#### 2. EIA (Energy Information Administration)
**Provider**: U.S. Department of Energy  
**Website**: https://www.eia.gov/  
**Data Used**:
- Electricity generation by source
- Energy consumption by sector
- Natural gas prices and production
- Oil prices and refinery capacity
- Renewable energy capacity
- State-level energy data

**API Registration**: https://www.eia.gov/opendata/register.php  
**Rate Limit**: 5,000 calls per hour  
**Coverage**: Real-time and historical energy market data

#### 3. BEA (Bureau of Economic Analysis)
**Provider**: U.S. Department of Commerce  
**Website**: https://www.bea.gov/  
**Data Used**:
- National Income and Product Accounts (NIPA)
- Regional economic accounts
- International trade data
- Industry-specific GDP contributions
- Personal income by state
- Fixed asset tables

**API Registration**: https://apps.bea.gov/API/signup/  
**Rate Limit**: 100 calls per hour  
**Coverage**: Comprehensive economic accounting data

#### 4. BLS (Bureau of Labor Statistics)
**Provider**: U.S. Department of Labor  
**Website**: https://www.bls.gov/  
**Data Used**:
- Employment statistics by industry
- Labor productivity data
- Occupational employment statistics
- Producer price indices
- Consumer expenditure surveys
- Regional employment data

**API Registration**: https://www.bls.gov/developers/  
**Rate Limit**: 500 calls per day  
**Coverage**: Detailed labor market and pricing data

### API Key Configuration

**IMPORTANT**: All data is sourced from authoritative government APIs for accuracy and reliability.

Create these API key files in the `secrets/` directory:

```bash
mkdir -p secrets/
echo "your_fred_api_key_here" > secrets/FRED_API_KEY.txt
echo "your_eia_api_key_here" > secrets/EIA_API_KEY.txt
echo "your_bea_api_key_here" > secrets/BEA_API_KEY.txt
echo "your_bls_api_key_here" > secrets/BLS_API_KEY.txt
```

**File Structure**:
```
secrets/
├── FRED_API_KEY.txt    # Federal Reserve Economic Data
├── EIA_API_KEY.txt     # Energy Information Administration
├── BEA_API_KEY.txt     # Bureau of Economic Analysis
├── BLS_API_KEY.txt     # Bureau of Labor Statistics
└── OPENAI_API_KEY.txt  # OpenAI for LLM analysis
```

### Data Integration Features

**Real-Time Updates**: Data is cached with 6-hour TTL and automatically refreshes  
**Quality Scoring**: Each data series receives a quality score based on:
- Data completeness (missing values)
- Update frequency consistency
- Historical coverage depth
- Source reliability

**Intelligent Caching**: 
- Local disk cache to minimize API calls
- Cache invalidation based on data freshness
- Automatic retry logic for failed requests

**Regional Data**: State and regional-level data for localized policy analysis

### Climate Scenarios & Academic Sources
- **NGFS (Network for Greening the Financial System)**: Climate scenarios for central banks
- **IPCC AR6**: Intergovernmental Panel on Climate Change Assessment Report 6
- **Historical policy implementations**: Real-world policy outcomes
- **Academic literature**: Peer-reviewed policy impact studies

### Verification Commands

Test your API configuration:
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from climate_risk_scenario_generation.data.data_integrator import DataSourceIntegrator
integrator = DataSourceIntegrator()
for api in ['fred', 'eia', 'bea', 'bls']:
    config = integrator.data_sources[api]
    print(f'{api.upper()}: {\"✓\" if config.api_key else \"✗\"} API key configured')
"
```

**Expected Output**:
```
FRED: ✓ API key configured
EIA: ✓ API key configured  
BEA: ✓ API key configured
BLS: ✓ API key configured
```

## Testing

The system includes automated integration tests to verify all components work correctly.

### Running Tests

#### Unit Tests (Recommended)

Run comprehensive unit tests for all system components:

```bash
# Run all unit tests
python3 -m pytest tests/ -v

# Run specific test modules
python3 -m pytest tests/test_policy_parser.py -v
python3 -m pytest tests/test_climate_data.py -v
python3 -m pytest tests/test_generic_policy_model.py -v
python3 -m pytest tests/test_integrated_analyzer.py -v
python3 -m pytest tests/test_publication_figures.py -v

# Run with coverage report
python3 -m pytest tests/ --cov=src --cov-report=html
```

The unit test suite validates:
- ✅ Policy parameter parsing from natural language
- ✅ Climate data provider functionality
- ✅ Economic model calculations
- ✅ Integrated analysis workflows
- ✅ Visualization generation
- ✅ Error handling and edge cases

#### Integration Tests

Run end-to-end integration tests:

```bash
# Direct execution (recommended)
python3 tests/integration_test.py

# Or from project root
cd /path/to/climate_risk_scenario_generation
python3 tests/integration_test.py
```

The integration test suite validates:
- ✅ All API endpoints (health, models, scenarios, examples)
- ✅ Query processing with OpenAI integration
- ✅ Input validation and error handling
- ✅ Rate limiting functionality
- ✅ Static file serving
- ✅ Backend ↔ Frontend communication

### Test Requirements

**Unit Tests:**
- No external dependencies required
- Tests run in isolation with mock data
- Can be run offline

**Integration Tests:**
- Server must be running: `python3 start_ui.py`
- OpenAI API key configured
- Internet connection for API calls

## Limitations

### Model Limitations

1. **Simplified Representations**
   - Economic models use representative parameters
   - Cannot capture all real-world complexities
   - Regional variations may be approximated

2. **Data Constraints**
   - Limited historical data for some policy types
   - Projections based on current trends
   - Model calibration reflects historical patterns

3. **Uncertainty Factors**
   - Future technology developments unpredictable
   - Political and social factors not fully modeled
   - Climate feedback loops simplified

### LLM Limitations

1. **Interpretation Quality**
   - Depends on model capability and training data
   - May not capture nuanced policy interactions
   - Local models (Ollama) may be less capable than OpenAI

2. **Cost Considerations**
   - OpenAI models incur per-query costs
   - Complex queries consume more tokens
   - Costs can accumulate with extensive use

3. **Connectivity Requirements**
   - OpenAI models require internet connection
   - API rate limits may apply
   - Service availability dependent on provider

### System Limitations

1. **Query Parsing**
   - May misinterpret ambiguous queries
   - Limited to policy types in taxonomy
   - Complex multi-policy scenarios challenging

2. **Temporal Scope**
   - Focuses on 2025-2050 timeframe
   - Historical calibration may not reflect future
   - Long-term projections increasingly uncertain

3. **Geographic Coverage**
   - Primary focus on US policies
   - International coverage limited
   - Sub-national variations approximated

## Troubleshooting

### Common Situations

**"System not ready" Error**
- Check OpenAI API key in `secrets/OPENAI_API_KEY.txt` (should contain only the key, no quotes)
- Verify the API key is valid by testing it: `curl -H "Authorization: Bearer $(cat secrets/OPENAI_API_KEY.txt)" https://api.openai.com/v1/models`
- Ensure internet connection for OpenAI models
- Verify all dependencies installed: `pip install -r requirements.txt`
- Make sure virtual environment is activated

**Ollama Models Not Showing**
- Confirm Ollama server running: `ollama serve`
- Check models installed: `ollama list`
- Verify server accessible: `curl http://localhost:11434/api/tags`

**Slow Response Times**
- Try switching to faster models (GPT-3.5 Turbo or Ollama)
- Check internet connection stability
- Consider local Ollama models for faster response

**Visualization Errors**
- Ensure matplotlib backend supports display
- Check file permissions in `static/viz/` directory
- Verify seaborn and numpy versions compatible

**Memory Considerations**
- Close other applications to free RAM
- Use smaller Ollama models (8B instead of 13B)
- Restart the application periodically

**Import/Module Errors**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Verify you're in the correct directory (should contain `start_ui.py`)
- Check Python path: the `src` directory should be in your project root
- Try reinstalling dependencies: `pip install --force-reinstall -r requirements.txt`

**Port Already in Use**
- If you get "port 5000 already in use", kill existing processes: `lsof -ti:5000 | xargs kill -9`
- Or change the port by editing `start_ui.py` (change `port=5000` to another number)

**spaCy Model Download**
- If `python -m spacy download en_core_web_sm` fails, try the direct wheel install method shown in setup step 4
- Ensure you have write permissions to your Python environment
- Check internet connection and try again

### Getting Help

1. **Check Logs**
   - Review console output for error messages
   - Look for specific error codes or stack traces

2. **Verify Configuration**
   - Double-check API keys and file paths
   - Ensure all services are running (Ollama)

3. **Test Components**
   - Try simple queries first
   - Test both OpenAI and Ollama models separately
   - Verify individual system components

### Performance Tips

1. **Model Selection**
   - Use GPT-3.5 Turbo for routine analysis
   - Reserve GPT-4 for complex scenarios
   - Consider Ollama for batch processing

2. **Query Optimization**
   - Be specific and clear in queries
   - Avoid overly complex multi-part questions
   - Use example queries as templates

3. **Resource Management**
   - Close browser tabs when not needed
   - Restart application after extended use
   - Monitor system resources during operation

---

For additional support, please refer to the project documentation or contact the development team.
