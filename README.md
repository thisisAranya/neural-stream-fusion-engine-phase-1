## 🚀 Quick Start

### Prerequisites
```bash
# Install Python 3.11+
python3.11 --version

# Install Docker
docker --version
docker-compose --version
```

### Setup
```bash
# 1. Clone repository
git clone https://github.com/yourusername/neural-stream-fusion-engine.git
cd neural-stream-fusion-engine

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models
chmod +x scripts/setup_models.sh
./scripts/setup_models.sh

# 5. Configure environment
cp .env.example .env
# Edit .env if needed

# 6. Start the system
python main.py

# 7. Test the API
python scripts/test_api.py
```

### Docker Setup (Recommended)
```bash
# Start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Test the system
python scripts/test_api.py
```

### First API Call
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Hello, how are you?",
       "max_tokens": 100,
       "temperature": 0.7
     }'
```
