# 🚀 Deployment Guide

This guide provides step-by-step instructions for deploying the Employee Salary Prediction MVP to various platforms.

## 📋 Prerequisites

- All project files are ready
- Model files (`.pkl` files) are generated
- Dependencies are installed locally

## 🖥️ Local Development

### 1. Run Locally
```bash
# Navigate to project directory
cd salary

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### 2. Access the Application
- Open your browser
- Navigate to `http://localhost:8501`
- The application will be available locally

## ☁️ Streamlit Cloud (Recommended for MVP)

### 1. Prepare Your Repository
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Employee Salary Prediction MVP"

# Create a GitHub repository and push
git remote add origin https://github.com/yourusername/employee-salary-prediction.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to `app.py`
6. Click "Deploy"

### 3. Configuration
- **Python version**: 3.8+
- **Main file**: `app.py`
- **Requirements file**: `requirements.txt`

## 🐳 Docker Deployment

### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run
```bash
# Build the image
docker build -t salary-prediction-app .

# Run the container
docker run -p 8501:8501 salary-prediction-app
```

### 3. Docker Compose (Optional)
```yaml
# docker-compose.yml
version: '3.8'
services:
  salary-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🚀 Heroku Deployment

### 1. Create Required Files

**Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh:**
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

### 2. Deploy to Heroku
```bash
# Install Heroku CLI
# Create new app
heroku create your-salary-app-name

# Add buildpack
heroku buildpacks:add heroku/python

# Deploy
git push heroku main

# Open the app
heroku open
```

## ☁️ AWS/GCP Deployment

### 1. AWS Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.9 salary-prediction

# Create environment
eb create salary-prediction-env

# Deploy
eb deploy
```

### 2. Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/salary-prediction

# Deploy to Cloud Run
gcloud run deploy salary-prediction \
  --image gcr.io/PROJECT-ID/salary-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔧 Environment Variables

### Streamlit Configuration
```bash
# Create .streamlit/config.toml
[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
```

### Production Settings
```bash
# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_PORT=8501
```

## 📊 Monitoring and Logs

### 1. Streamlit Cloud
- Built-in monitoring dashboard
- Automatic scaling
- Error logs and analytics

### 2. Custom Monitoring
```python
# Add logging to app.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction made: {predicted_salary}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   ```

2. **Model Loading Errors**
   - Ensure all `.pkl` files are in the project directory
   - Check file permissions
   - Verify model compatibility

3. **Dependency Issues**
   ```bash
   # Clear pip cache
   pip cache purge
   
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

### Performance Optimization

1. **Model Caching**
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load('model.pkl')
   ```

2. **Data Caching**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv('data.csv')
   ```

## 🔒 Security Considerations

1. **Input Validation**
   - Validate all user inputs
   - Sanitize data before processing

2. **Rate Limiting**
   - Implement request throttling
   - Monitor usage patterns

3. **Error Handling**
   - Don't expose sensitive information in error messages
   - Log errors for debugging

## 📈 Scaling Considerations

1. **Horizontal Scaling**
   - Use load balancers
   - Deploy multiple instances

2. **Caching**
   - Redis for session storage
   - CDN for static assets

3. **Database**
   - Consider adding a database for user sessions
   - Implement user authentication if needed

## 🎯 Next Steps

After successful deployment:

1. **Monitor Performance**
   - Track response times
   - Monitor error rates
   - Analyze user behavior

2. **Iterate and Improve**
   - Collect user feedback
   - Improve model accuracy
   - Add new features

3. **Scale Up**
   - Add more sophisticated models
   - Implement user management
   - Add analytics dashboard

---

**Happy Deploying! 🚀**
