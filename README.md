# Dynamic Brand Health & Sentiment-Aware Recommendations: Bridging Unstructured Reviews and Structured Ratings

# Overview

This project implements a sentiment-aware hybrid recommendation system designed to optimize e-commerce product discovery by moving beyond static numerical ratings. By integrating a multi-stage architecture, the system utilizes Deep Learning (BERT) to extract emotional nuance from unstructured consumer reviews and Linear Regression to track brand health momentum across time. These sentiment-driven signals are then fused with a Collaborative Filtering (SVD) model to produce a final hybrid score, ensuring that recommendations are not only personally relevant based on historical behavior but also validated by objective brand quality and positive market trends.

<img width="1392" height="768" alt="Gemini_Generated_Image_4gnfot4gnfot4gnf" src="https://github.com/user-attachments/assets/c8535c6d-70cd-42f6-974c-84fff4c54a48" />

# Data Processing

**1. Handling of Missing Values**

  The initial phase addresses data sparsity to ensure robust model training and preventing system failures:

- reviews.text: Rows with missing text are dropped because this field is the core requirement for sentiment analysis. With only 36 missing values, the loss is statistically negligible.

- reviews.title: Instead of dropping these rows, they are filled with "No Title" to preserve secondary context without losing the data entry.

- reviews.username: These are filled with "Anonymous" because the recommendation system requires a user identifier to group interactions effectively.

- reviews.didPurchase: Since over 50% are missing, this is filled with "Unknown". This treats the lack of information as a separate category, which is safer than dropping rows or guessing values.

- reviews.doRecommend: This field is imputed with the Mode (the most frequent choice). This is useful for validating the sentiment analysis model later.

**2. Feature Engineering & Column Rationalization**

This step optimizes the dataset by creating high-value features and removing "noise":

- Merge Text Fields: The reviews.title and reviews.text are combined into a full_review column to capture all possible sentiment signals in one place.

- Sentiment Labeling: A new target variable is created where ratings of 4-5 are "Positive", 3 is "Neutral", and 1-2 are "Negative".

- Dropping Redundant Columns: Columns like ean, upc, keys, and manufacturerNumber are dropped as they are redundant for the specific goal of text-based analysis.

- Geographic Pruning: userCity and userProvince are dropped due to extremely high null counts (90%+), making them statistically useless for this scope.

- Temporal Pruning: dateAdded and dateUpdated are removed because reviews.date provides a more accurate timeline for analyzing consumer trends.

**3. Advanced Text Preprocessing**

To prepare the unstructured text for a Transformer-based model (BERT), it must be normalized:

- Standardization: Text is converted to lowercase and regex is used to remove punctuation, special characters, and numbers to ensure the model focuses on semantic meaning.

- Stopword Removal: Common English words (e.g., "the", "is") are removed as they do not carry emotional weight or sentiment.

- Lemmatization: Words are reduced to their dictionary root form (e.g., "running" to "run") to normalize the vocabulary and reduce feature space.

**4. Transformation for Deep Learning**

The final stage prepares the data specifically for the BERT model and the recommendation engine:

- Vectorization: The processed text is transformed into numerical format using Token IDs and Attention Masks, allowing the BERT model to "read" the input.

- IOB Tagging (NER): A custom function generates IOB tags (B-BRAND, I-BRAND, B-CAT) by matching tokens in the review text to the brand and category metadata. This allows the model to identify specific entities within a review.

- Normalization: Numerical features such as reviews.numHelpful are often scaled using a MinMaxScaler to ensure they act as fair weights in the recommendation engine without overpowering the sentiment score.


# BERT Model Training: Named Entity Recognition (NER)

The training of the BERT model for Named Entity Recognition (NER) was a pivotal step in bridging the gap between raw, unstructured review text and structured data insights. This process transformed simple text strings into actionable intelligence about brands and product categories.

**1. Why it was Necessary**

While the dataset contained structured columns like brand and categories, consumer reviews often mention products, competitors, or specific sub-categories informally (e.g., "I usually buy Nike, but this brand is better").

- Entity Extraction: Traditional string matching is brittle. We needed a model that understands context to identify exactly which words in a review refer to a "Brand" versus a "Category".

- Noise Reduction: Reviews are filled with irrelevant text. The NER model allows the system to ignore the "fluff" and focus exclusively on the entities that drive consumer affinity.

- Validation: It serves as a cross-verification tool to ensure that the review text actually aligns with the metadata provided in the structured brand column.

**2. How It Was Done**

The training utilized Transfer Learning by fine-tuning a pre-trained bert-base-uncased model for token classification.

- Architecture: Used BertForTokenClassification, which adds a linear layer on top of the BERT hidden states to predict a label (B-BRAND, I-BRAND, B-CAT, I-CAT, or O) for every single token.

- Optimization: Employed the AdamW optimizer, a standard for fine-tuning Transformers to prevent aggressive weight updates that could destroy pre-trained knowledge.

- Data Preparation: Using the ReviewDataset class, we aligned our custom IOB (Inside-Outside-Beginning) tags with BERT’s sub-word tokens. This ensured that if a brand name was split into multiple tokens (e.g., "HealthKart" into "Health" and "##Kart"), the model learned to label the entire sequence correctly.

- Training Loop: The model was trained over 5 epochs, during which it learned to minimize the Cross-Entropy Loss between its predictions and our ground-truth IOB tags.

**3. What was Achieved**

- Contextual Intelligence: The model achieved the ability to distinguish between identical words used in different contexts (e.g., distinguishing "Apple" as a brand from "apple" as a fruit).

- High Precision Mapping: By saving the model and tokenizer, we created a reusable asset that can process thousands of new reviews per second, identifying brands and categories with high accuracy.

- Structured Unstructured Data: We successfully turned "dead" text into a "live" map of entity mentions.

**4. How it will be Used**

The trained NER model is the "front-end" of the intelligence pipeline:

- Automated Tagging: When a new review is submitted, the NER model automatically tags mentioned brands. These tags are then paired with the Sentiment Score to see how the user feels about that specific entity.

- Competitor Analysis: If a user mentions a competitor brand in a review, the system can identify it, allowing the brand to understand its market position relative to others.

- Refined Recommendations: By knowing exactly which "Categories" a user talks about most (e.g., "Protein Powders" vs "Multivitamins"), the recommendation engine can prioritize those specific categories in the user's feed.

# Extraction of Sentiment Weights using Pre-trained BERT

The integration of the nlptown/bert-base-multilingual-uncased-sentiment model allowed the system to move beyond binary "Positive/Negative" labels, capturing the specific intensity of consumer emotions expressed in the review text.

**1. Leveraging a Specialized Pre-trained Model**

Instead of training a sentiment classifier from scratch, we utilized a model specifically fine-tuned on a massive dataset of multilingual reviews.

- Star-Rating Alignment: This specific BERT variant is trained to predict sentiment on a 5-star scale (1 to 5 stars), which perfectly aligns with the structured rating system in our dataset.

- Multilingual Robustness: Since consumer reviews can contain diverse linguistic patterns, the multilingual base ensures that the model captures emotional context accurately even if the grammar is informal or mixed.

**2. The Sentiment-to-Weight Transformation Process**

The run_batch_sentiment function converts raw text into a mathematical weight using the following technical steps:

- Softmax Probability Extraction: The model outputs raw "logits" for each of the five star categories. By applying the Softmax function, we transform these logits into a probability distribution where the sum equals 1.0 (e.g., a review might be 10% likely to be 4-star and 80% likely to be 5-star).

- The Weighted Scale: We define a linear scale [0.2, 0.4, 0.6, 0.8, 1.0] representing the weight of each star rating (1 to 5).

- Calculating the Sentiment Weight: Using a matrix multiplication (torch.matmul), we multiply the star probabilities by the scale.

- Example: If a review has a 100% probability of being 5-star, its weight is 1.0. If it is 100% likely to be 3-star, its weight is 0.6.

This provides a continuous decimal score (e.g., 0.87) that captures subtle nuances—such as a review that is "mostly positive but with slight hesitation"—which a simple 5-star integer cannot.

**3. Why This Approach is Superior (Nuance vs. Noise)**

- Correcting Rating Bias: Consumers often give a "5-star" rating even when their text indicates minor complaints. The BERT sentiment weight looks at the actual words and can pull that score down to a 0.82, providing a more "honest" reflection of brand quality.

- Continuous Input for Brand Health: Unlike categorical labels, these decimal weights are easily averaged to create the Brand Health Score, ensuring that the 50% weight assigned to sentiment is mathematically precise.

**4. Integration into the Pipeline**

- Brand Health: The average of these weights across all reviews for a brand becomes the primary signal for its health score.

- Hybrid Recommender: These weights serve as the "Sentiment Filter." If two products have similar predicted ratings from the SVD model, the one with the higher BERT Sentiment Weight will be ranked higher in the user's feed.

# Brand Health KPI Analysis

The Brand Health Lookup serves as the system's strategic "Key Performance Indicator" (KPI). It translates thousands of unstructured emotional signals and structured ratings into a single, standardized score for every brand in the database. This allows the recommendation engine to prioritize products from brands that are objectively healthy and consumer-favored.

**1. The Weighted Brand Health Formula**
   
Instead of relying on a simple average of star ratings—which can be easily skewed—we implement a Sentiment-Aware Composite Formula:

$$Brand\ Health\ Score = \left( \overline{S}_{weight} \times 0.5 \right) + \left( \frac{\overline{R}}{5} \times 0.3 \right) + \left( \overline{Rec} \times 0.2 \right)$$

Where: $\overline{S}_{weight}$: The average BERT Sentiment Weight for the brand (0 to 1).

$\overline{R}$: The average Numerical Rating (Scale 1–5), normalized by dividing by 5.

$\overline{Rec}$: The average Recommendation Rate (percentage of "True" responses in reviews.doRecommend).

The final result is multiplied by 100 to produce a robust score between 0 and 100.

**2. Why This Formula Works**

- Nuance vs. Noise (The 50% Sentiment Anchor)
Star ratings are often extreme—users typically leave either a 1 or a 5. By giving 50% weight to BERT Sentiment, we prioritize the actual emotional nuance found in the review text. This is far more reliable than a simple click because the BERT model detects if a "5-star" review actually contains hidden complaints, effectively pulling the health score down to a more realistic level.

- Bias Correction & False Positive Prevention
If a user gives a high rating but writes a negative review (a "false positive"), the BERT weight acts as a corrective filter. This prevents the recommendation engine from suggesting products that are currently suffering from quality issues that haven't yet reflected in the average star rating.

- Capturing Consumer Intent
The inclusion of the reviews.doRecommend signal (20% weight) captures loyalty. A brand might have decent sentiment, but if users are explicitly saying they wouldn't recommend it to others, the brand's "health" is technically lower. This acts as a final validation of the product’s real-world value.

- Unified Normalization
By dividing the star rating by 5 and using the probability of recommendation (0 to 1), all three disparate signals are placed on the same mathematical scale. This ensures that no single metric accidentally overpowers the others, resulting in a balanced and fair assessment of every brand.

**3. Implementation** 

As seen in the BrandHealthEngine class, the system aggregates these three key signals to output a ranked lookup table:

- Step 1: It finds the average BERT Sentiment Weight for the brand.

- Step 2: It calculates the average numerical rating.

- Step 3: It determines the percentage of "Yes" recommendations.

- Step 4: It applies the weighted formula to combine them into one final Health Score.

# Identifying Brand Market Trends

While the Brand Health Score provides a static "snapshot" of quality, identifying Brand Trends adds a dynamic, predictive layer to the system. By analyzing how health scores fluctuate over time, the system can distinguish between "Legacy" brands in decline and "Rising Stars" gaining consumer favor.

**1. The Trend Identification Pipeline**

The system uses Linear Regression to quantify the "momentum" of a brand’s health across specific time intervals (e.g., monthly or weekly).

- Robust Date Parsing: The process begins by standardizing the reviews.date field using mixed-format parsing and UTC offsets to handle diverse timestamp styles. Rows with invalid or missing dates are removed to ensure time-series integrity.

- Time Bucketing: Reviews are grouped into discrete time_buckets (e.g., 2024-01, 2024-02). This allows the system to calculate a distinct Brand Health Score for every month in the brand's history.

- Regression Analysis: For every brand with at least two data points, the system fits a Linear Regression model where:

- X (Independent Variable): The chronological sequence of time buckets.

- y (Dependent Variable): The calculated Brand Health Score for that period.

**2. Slope-Based Trend Classification**

The most critical output of this process is the Regression Slope, which indicates the velocity and direction of the brand's sentiment. We classify brands into three statuses based on this slope:

<img width="1536" height="1024" alt="ChatGPT Image Jan 1, 2026, 06_54_05 PM" src="https://github.com/user-attachments/assets/32494ce1-c8df-472e-902c-5553f2bacb88" />


**3. Strategic Value of Trend Identification**

- Predictive Advantage: Instead of just knowing a brand is good, the system identifies if a brand is getting better. This allows the engine to "boost" products that are currently trending in the market.

- Filtering Declining Quality: If a high-rated legacy brand has a Downward trend status, the system can proactively penalize its rank in the recommendation feed, preventing users from receiving products that are no longer meeting historical standards.

- Actionable Business Insights: For stakeholders, the slope and trend_status provide a "Brand Pulse" that highlights which manufacturers are successfully capturing consumer interest.

# The Hybrid Recommendation Engine

The final step of the project integrates all previous phases—Collaborative Filtering, Deep Learning Sentiment Analysis, and Regression-based Trend Tracking—into a single Two-Stage Re-ranking Architecture. This engine ensures that recommendations are not only personalized but also high-quality and trending.

**1. The Hybrid Scoring Formula**

To determine the final ranking of products, the system applies a weighted formula that balances three distinct signals:

$$Hybrid\ Score = \left( CF\ Score \times 0.8 \right) + \left( Health\ Score\ Factor \times 0.1 \right) + \left( Trend\ Boost \times 0.1 \right)$$

- CF Score (80%): The "Predicted Rating" from the SVD (Singular Value Decomposition) model. This remains the primary driver, ensuring the suggestion matches the user's personal historical taste.

- Health Score Factor (10%): Derived from the Brand Health Lookup. It acts as a quality filter, boosting products from brands with verified positive sentiment and high recommendation rates.

- Trend Boost (10%): Derived from the Market Trend Regression. It provides a final "nudge" for brands with Upward momentum and penalizes those with Downward slopes.

**2. Handling the "Cold Start" Problem**

A critical feature of this final step is how it handles new users who have no historical data (the Cold Start problem):

- Existing Users: The system loops through all products, calculates the hybrid score, and returns the top N suggestions.

- New Users: The engine bypasses Collaborative Filtering and instead recommends "Rising Stars"—products from brands that currently hold the highest combined Health Score and Upward Trend Slope. This ensures every user gets high-quality suggestions immediately.

**3. Results and Impact**

As shown in the output, the system successfully identifies multi-category interests for users like "Dorothy W" while maintaining high quality standards:

- Filtering: It filters out brands with Stable or Downward trends, ensuring users are only suggested products gaining consumer favor.

- Nuanced Discovery: Even items without major brand names (labeled "No Brand") can rank highly if the BERT Sentiment Task detects high-quality sentiment in their raw reviews.

<img width="1201" height="690" alt="dorothy" src="https://github.com/user-attachments/assets/54fc3a32-5306-456e-84ca-15f50b769b05" />


# How to Run

**1. Clone the Repository**

git clone (https://github.com/neomotion/Dynamic-Brand-Health-Sentiment-Aware-Recommendations.git)

cd Dynamic-Brand-Health-Sentiment-Aware-Recommendations

**2. Rebuild the Image (Environment Preparation)**

Since Docker images themselves are usually too large for Git, the user will "re-materialize" the image on their machine using your Dockerfile.

docker build -t health-recommender .

**3. Run the Container**

docker run -it health-recommender

# Conclusion

The conclusion of this project marks the successful development of a robust, "sentiment-aware" hybrid recommendation system that bridges the gap between structured interaction data and unstructured consumer feedback. By leveraging a multi-stage architecture, the system employs BERT-based Deep Learning to extract nuanced emotional weights from reviews and Linear Regression to quantify brand health momentum over time. These sophisticated signals are integrated with an SVD-based collaborative filtering model through a weighted hybrid scoring formula, ensuring that product suggestions are not only personally relevant but also objectively high-quality and market-aligned. Finally, the project’s deployment through Docker provides a portable, scalable solution that addresses critical challenges like the "Cold Start" problem, ultimately offering a more trustworthy and dynamic discovery experience for e-commerce users.
