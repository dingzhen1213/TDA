### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/dingzhen1213/TDA.git
cd TDA

# Install dependencies
pip install -r requirements.txt
### 2. Dataset Preparation

1. **PAMAP2 Dataset**:
   - Download the PAMAP2 dataset from the official source
   - Extract the `Protocol` folder to `datasets/pamap2_dataset/` directory

2. **UCI-HAR Dataset**:
   - Download the UCI-HAR dataset from the UCI Machine Learning Repository
   - Extract all files to `datasets/UCI HAR Dataset/` directory

3. **WISDM Dataset**:
   - Download the WISDM dataset
   - Place `WISDM_ar_v1.1_raw.txt` in `datasets/WISDM_Dataset/` directory

4. **Configuration**:
   - Update the data paths in configuration files to match your dataset locations

### 3. Model Training

Run the following commands to train and evaluate models on each dataset:

```bash
# Train and evaluate on PAMAP2 dataset
python main_pamap2.py --train --eval

# Train and evaluate on UCI-HAR dataset  
python main_UCI.py --train --eval

# Train and evaluate on WISDM dataset
python main_WISDM.py --train --eval
