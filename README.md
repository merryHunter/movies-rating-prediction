# Movie rating prediction Solution

Files with solutions for 3 modalities:

1. `numerical.ipynb` - EDA, Catboost gradient boosting solution. Using only `movies.csv` without extra feature engineering.
2. `text.ipynb` - Using SentenceTransformers to build embeddings on top of concatenated tagline + desription.
3. `poster_image.ipynb` - Using SigLIP model to generate embeddings and training MLP on top of vectors with MSE and LogCosh.

Part of stacking input features to train one model or voting was skipped.

Basic models individually showed weak prediction capabilities, RMSE for rating ranging from 0.50 to 0.


### Implementation plan

1. EDA, understand the task - 20 minutes
2. Train/ test split for evaluation 10 min
3. Build simple numerical categorical model - xgboost 20 min
4. Evaluate categorical model 10 min - use RMSE and MAE a
5. Build text based model - use transformers to generate embeddings
6. Buid image based model - use transformers to generate embeddings
7. Use simple MLP with MSE and logcosh loss function
8. skipped - Voting for ensemble or stack features into 1 input vector