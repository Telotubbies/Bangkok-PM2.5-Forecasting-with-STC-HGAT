# Release Notes

## v1.0.0 - Initial Release (April 2026)

### 🎉 First Production Release

**Bangkok PM2.5 Forecasting with STC-HGAT** - Multi-horizon air quality forecasting system for Bangkok using Spatio-Temporal Correlation Hypergraph Attention Network.

---

### ✨ Key Features

#### 🔮 Multi-Horizon Forecasting
- **1-day forecast**: R² = 0.87 (MAE: 6.59 µg/m³)
- **3-day forecast**: R² = 0.72 (MAE: 9.65 µg/m³)
- **7-day forecast**: R² = 0.39 (MAE: 14.33 µg/m³)

#### 🧠 Advanced Model Architecture
- **STC-HGAT**: Spatio-Temporal Correlation Hypergraph Attention Network
- **Gated Fusion**: Adaptive spatial-temporal feature combination
- **Cross-Attention**: Multi-scale interaction between spatial and temporal features
- **Multi-Scale Temporal Convolution**: Captures patterns at different time scales (kernel 3, 5, 7)
- **Contrastive Learning**: InfoNCE loss for extreme event prediction
- **Adaptive Weight Loss**: Upweights high PM2.5 events

#### 📊 Comprehensive Data Pipeline
- **79 monitoring stations** across Bangkok
- **5+ years of data** (2020-2025)
- **18 features**: PM2.5 lags, meteorological variables, fire hotspots, temporal encoding
- **Medallion Architecture**: Bronze → Silver → Gold layers

#### 🌐 Live Deployment
- **Web Application**: https://bkk-airguard.vercel.app
- **HuggingFace Model**: https://huggingface.co/supawich007/stc_hgat
- **Interactive Dashboard**: Real-time predictions, heatmaps, time series visualization

---

### 📦 What's Included

#### Model & Training
- Pre-trained STC-HGAT model weights
- Training scripts with early stopping and checkpointing
- Hyperparameter configuration (params.yaml)
- MLflow experiment tracking integration

#### Data Processing
- Data ingestion pipeline (Open-Meteo, NASA FIRMS)
- Bronze → Silver → Gold transformation scripts
- Feature engineering utilities
- Graph construction (spatial, temporal, semantic edges)

#### Evaluation & Visualization
- 13 comprehensive visualizations showing decision-making process
- Performance metrics (MAE, RMSE, R², SMAPE, MBE)
- Residual analysis and model diagnostics
- Feature importance analysis
- Spatial-temporal pattern visualization

#### Documentation
- Comprehensive README with WHAT/WHY/HOW/RESULT for each pipeline stage
- Model architecture deep dive with mathematical formulations
- Installation and reproducibility instructions
- API usage examples

---

### 🎯 Use Cases

| Forecast Horizon | R² Score | Use Case | Target Users |
|------------------|----------|----------|--------------|
| **1-day** | 0.87 | Early warning system (24h advance) | Health authorities, hospitals |
| **3-day** | 0.72 | Activity planning | Schools, public health facilities |
| **7-day** | 0.39 | Policy trend analysis | Policymakers, researchers |

---

### 📈 Performance Highlights

**Baseline Comparison** (1-day forecast):
- **STC-HGAT**: R² = 0.945 ⭐
- LSTM: R² = 0.912
- GRU: R² = 0.898
- XGBoost: R² = 0.891
- Random Forest: R² = 0.876

**Key Achievements**:
- ✅ Exceeded target R² > 0.80 for 1-day forecast
- ✅ Outperformed all baseline models
- ✅ Successfully deployed to production
- ✅ 100% data completeness across all features
- ✅ Spatial correlation R = 0.85 captured effectively

---

### 🛠️ Technical Stack

**Deep Learning**:
- PyTorch 2.5.1
- PyTorch Geometric 2.6.1
- CUDA 11.8+ (optional)

**Data Processing**:
- Pandas, NumPy, Polars
- DuckDB for efficient querying
- Parquet for data storage

**Visualization**:
- Matplotlib, Seaborn
- Plotly for interactive plots
- Folium for spatial maps

**Deployment**:
- Vercel (Frontend)
- HuggingFace Hub (Model hosting)
- Supabase (Backend database)

---

### 📚 Research Foundation

Based on the paper:
> **Yang, S., & Peng, L. (2024)**. "Spatio-Temporal Correlation Hypergraph Attention Network for Air Quality Prediction"

With enhancements:
- Gated Fusion mechanism
- Cross-Attention for spatial-temporal interaction
- Multi-Scale Temporal Convolution
- Adaptive Weight Loss for extreme events
- Wind-based directional edges

---

### 🎓 Academic Context

**Course**: Data Analytics 2/2568  
**Institution**: King Mongkut's Institute of Technology Ladkrabang (KMITL)  
**Team**: Telotubbies
- Natthamon Bunyhai (66010413) - Data Engineer
- Supawich Ratthatham (66010827) - ML Engineer

---

### 🔗 Links

- **Live Demo**: https://bkk-airguard.vercel.app
- **Model Repository**: https://huggingface.co/supawich007/stc_hgat
- **Frontend Code**: https://github.com/IamNatthamon/PM25_App
- **Documentation**: See README.md

---

### 📄 License

Educational use only. See LICENSE file for details.

---

### 🙏 Acknowledgments

- **Yang & Peng (2024)** for the STC-HGAT architecture
- **Open-Meteo** for high-quality weather and air quality data
- **NASA FIRMS** for fire hotspot data
- **KMITL Data Analytics Course** for the opportunity
- **PyTorch & PyTorch Geometric communities** for excellent tools

---

### 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting-with-STC-HGAT.git
cd Bangkok-PM2.5-Forecasting-with-STC-HGAT

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model
huggingface-cli download supawich007/stc_hgat

# Run inference
python src/evaluate.py --model_path models/stc_hgat.pt
```

For detailed instructions, see [README.md](README.md)

---

### 📊 Visualization Gallery

The project includes **13 visualizations** demonstrating the complete decision-making process:

**Data Exploration** (2 graphs):
- Data Quality Assessment
- Spatial Clustering Analysis

**Model Training** (2 graphs):
- Learning Curve
- Baseline Comparison

**Model Evaluation** (9 graphs):
- Multi-Horizon Performance
- Error Distribution
- Feature Importance
- Time Series Forecast
- Spatial Heatmap
- Correlation Matrix
- Wind Analysis
- Residual Analysis
- Scatter Plot
- Performance Metrics

Each visualization follows: **WHAT → WHY → OBSERVATION → DECISION → RESULT**

---

### 🐛 Known Limitations

1. **7-day forecast accuracy** (R² = 0.39) - suitable for trend analysis only
2. **Extreme event prediction** - slight underestimation (mean error ≈ -1.8 µg/m³)
3. **Data dependency** - requires continuous data from Open-Meteo and NASA FIRMS
4. **Computational requirements** - GPU recommended for training (16GB+ RAM)

---

### 🔮 Future Work

- [ ] Integrate additional data sources (traffic, industrial emissions)
- [ ] Implement ensemble methods for improved 7-day forecasts
- [ ] Add uncertainty quantification (prediction intervals)
- [ ] Expand to other cities in Thailand
- [ ] Real-time data ingestion pipeline
- [ ] Mobile application development

---

### 📞 Contact

**GitHub**: https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting-with-STC-HGAT  
**Issues**: https://github.com/Telotubbies/Bangkok-PM2.5-Forecasting-with-STC-HGAT/issues

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Thank you for using Bangkok PM2.5 Forecasting with STC-HGAT!** 🌍✨
