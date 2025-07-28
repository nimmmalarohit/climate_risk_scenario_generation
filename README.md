# Climate Risk Scenario Generation

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
   pip install --upgrade pip
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
2. [Setup Instructions](#setup-instructions)
3. [OpenAI and Ollama Configuration](#openai-and-ollama-configuration)
4. [Parameter Tuning](#parameter-tuning)
5. [Launching the UI](#launching-the-ui)
6. [Navigating the Interface](#navigating-the-interface)
7. [Understanding Results](#understanding-results)
8. [Data Sources](#data-sources)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)

## System Overview

The Climate Risk Scenario Generation system analyzes "what-if" climate policy scenarios using real quantitative economic models combined with language model interpretation. 

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

3. **Upgrade pip and install dependencies**
   ```bash
   pip install --upgrade pip
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
3. **You should see** a blue gradient page with "Climate Risk Scenario Generation" at the top
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

The system integrates data from multiple authoritative sources:

### Climate Scenarios
- **NGFS (Network for Greening the Financial System)**: Climate scenarios for central banks
- **IPCC AR6**: Intergovernmental Panel on Climate Change Assessment Report 6

### Economic Data
- **Federal Reserve Economic Data (FRED)**: US economic indicators
- **World Bank**: Global economic and development data
- **IEA (International Energy Agency)**: Energy sector data

### Policy Data
- **Historical policy implementations**: Real-world policy outcomes
- **Academic literature**: Peer-reviewed policy impact studies
- **Government databases**: Official policy documentation

### Model Calibration
- **Sectoral elasticities**: From econometric studies
- **Technology learning curves**: Historical technology adoption data
- **Carbon pricing effects**: Real-world carbon market data

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

### Common Issues

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

**Memory Issues**
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

**spaCy Model Download Issues**
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

For additional support or to report issues, please refer to the project documentation or contact the development team.
