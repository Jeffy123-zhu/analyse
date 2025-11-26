# NFL Player Trajectory Prediction

Physics-based trajectory prediction system for NFL Big Data Bowl 2026.

## Demo

Open `index.html` in your browser to view the interactive results dashboard.

## Performance

**Validation RMSE:** 4.17 yards (50.5% improvement over baseline)

| Model | RMSE | Rank |
|-------|------|------|
| Baseline | 8.43 yards | Top 40-50% |
| **Improved** | **4.17 yards** | **Top 25-35%** |

## Quick Start

```bash
python improved_predictor.py
```

Output: `submissions/improved_submission.csv` (99,266 predictions)

## Files

```
nfl-trajectory-prediction/
├── improved_predictor.py       Main prediction model (308 lines)
├── index.html                  Interactive results dashboard
├── submissions/
│   └── improved_submission.csv 
└── README.md                   
```

## Algorithm

### Two Prediction Methods

**1. Acceleration-Based** (for general movement)
- Uses current velocity and acceleration
- Applies drag coefficient (0.98)
- Enforces max speed (12 yards/sec)

**2. Ball-Oriented** (when ball position known)
- Calculates direction to ball
- Applies role-based weighting:
  - Targeted Receiver: 70%
  - Defensive Coverage: 50%
  - Other: 30%
- Blends with current velocity

### Post-Processing

- Moving average smoothing (window=3)
- Field boundary enforcement
- Physics constraints

## Validation Results

| Week | Baseline | Improved | Improvement |
|------|----------|----------|-------------|
| 13 | 8.68 yards | 4.23 yards | 51.3% |
| 14 | 8.70 yards | 4.29 yards | 50.7% |
| 15 | 7.91 yards | 3.99 yards | 49.6% |
| **Avg** | **8.43 yards** | **4.17 yards** | **50.5%** |

## Implementation

**Language:** Pure Python 3.8+  
**Dependencies:** None (csv, math, pathlib only)  
**Runtime:** ~2 minutes  
**Memory:** < 500 MB  
**Code:** 308 lines

## Usage

### Run Prediction

```bash
python improved_predictor.py
```

### View Results

Open `index.html` in browser for interactive dashboard.

### Customize Parameters

Edit `improved_predictor.py`:

```python
# Line 42: Max speed
if speed > 12:  # yards/second

# Line 52-56: Ball attraction weights
blend = 0.7  # Targeted Receiver
blend = 0.5  # Defensive Coverage
blend = 0.3  # Other

# Line 97: Drag coefficient
vx *= 0.97
```

## Deployment

### Local

1. Clone repository
2. Run `python improved_predictor.py`
3. Open `index.html` in browser

### GitHub Pages

1. Push to GitHub
2. Enable GitHub Pages
3. Set source to main branch
4. Access at `https://yourusername.github.io/repo-name`

## Kaggle Submission

Upload `submissions/improved_submission.csv` to competition page.

## Technical Details

### Input Features
- Position: x, y coordinates
- Velocity: speed, direction
- Acceleration: a
- Orientation: o
- Ball landing: ball_land_x, ball_land_y
- Player role: Targeted Receiver, Defensive Coverage, Other

### Core Functions
```python
predict_with_acceleration()  # Acceleration-based prediction
predict_toward_ball()        # Ball-oriented prediction
smooth_trajectory()          # Moving average smoothing
process_week()               # Process weekly data
calculate_rmse()             # Evaluate performance
main()                       # Full pipeline
```

### Algorithm Complexity
- Time: O(n × m) where n=players, m=frames
- Space: O(n × m)

## Limitations

Current approach:
- Pure Python (no deep learning)
- Simple feature engineering
- Deterministic predictions
- Performance ceiling ~4.0 yards

To reach top 10%:
- LSTM/Transformer models
- 50+ engineered features
- Model ensembles
- Target RMSE < 3.0 yards

## License

MIT License

## Acknowledgments

Data from NFL Big Data Bowl 2026 competition on Kaggle.
