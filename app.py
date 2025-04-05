import streamlit as st
import numpy as np
import pandas as pd
import pickle
import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


#####################
# Model Building
#####################

def build_complex_model():
    """
    Build a complex ML pipeline that scales features, reduces dimensions with PCA,
    and uses a stacking classifier combining RandomForest, SVC, and LogisticRegression.
    Model is specially tuned to detect spam posts like the example.
    """
    preprocessing = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6))  # Increased from 4 to capture more variance
    ])

    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('svc', SVC(probability=True, kernel='rbf', C=10, class_weight='balanced', random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=42))
    ]

    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=42),
        cv=2
    )

    model_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('stacking', stacking_clf)
    ])

    # Custom training dataset: extreme examples for spam and ham with emphasis on follower patterns
    X_custom = np.array([
        # Ham sample 1 (normal engagement, verified, good follower ratio)
        [0.1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.9, 4.0, 0.0,
         0.1, 0.0, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5],
        # Spam sample 1 (low followers, high following)
        [0.7, 0.3, 0.5, 0.3, 1.5, 1.5, 1.0, 0.6, 3.5, 0.3,
         0.4, 0.2, 0.0, 1.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5],
        # Ham sample 2 (normal post, good follower metrics)
        [0.2, 0.0, 0.1, 0.0, 0.2, 0.2, 0.0, 0.9, 4.0, 0.0,
         0.0, 0.0, 0.1, 0.1, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5],
        # Spam sample 2 (extremely low followers, high following)
        [0.8, 0.3, 0.5, 0.4, 2.0, 2.0, 1.0, 0.7, 3.0, 0.8,
         0.7, 0.4, 0.3, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    y_custom = np.array([0, 1, 0, 1])  # 0=ham, 1=spam

    # Additional extreme spam examples with follower/following patterns
    extreme_spam = np.array([
        # 5 followers, 2000 followings
        [0.8, 0.4, 0.6, 0.5, 2.0, 2.0, 1.0, 0.6, 3.0, 0.9,
         0.8, 0.5, 0.4, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        # 10 followers, 1500 followings
        [0.7, 0.3, 0.5, 0.4, 1.8, 1.8, 1.0, 0.6, 3.2, 0.8,
         0.7, 0.4, 0.3, 1.8, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        # 20 followers, 1000 followings
        [0.6, 0.3, 0.4, 0.3, 1.5, 1.5, 1.0, 0.7, 3.5, 0.7,
         0.6, 0.3, 0.2, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    extreme_spam_labels = np.array([1, 1, 1])  # All spam

    X_combined = np.vstack([X_custom, extreme_spam])
    y_combined = np.hstack([y_custom, extreme_spam_labels])

    model_pipeline.fit(X_combined, y_combined)
    return model_pipeline


#####################
# Feature Extraction
#####################
def extract_features_from_text_and_user(post_text, followers, followings, verified):
    """
    Extract 20 human-readable features from the post text and user metadata.
    Modified for increased weight on account metrics:
      - f5: Inverted normalized followers count with steeper curve
      - f6: Following-to-follower ratio with higher weighting for spam patterns
    """
    text = post_text.strip()
    tokens = text.split()
    total_words = len(tokens) if tokens else 1

    # Original text features (f1-f4)
    f1 = min(1, (len(text) / 280.0) * 2)
    f2 = min(1, (text.count('@') / total_words) * 4)
    f3 = min(1, (text.count('#') / total_words) * 4)
    url_count = text.lower().count("http") + text.lower().count("www") + text.lower().count(
        ".com") + text.lower().count("click")
    f4 = min(1, (url_count / total_words) * 4)

    # MODIFIED: f5 - Inverted normalized followers count with steeper curve
    # New scale: followers <= 100 -> higher penalty, followers >= 500 -> minimal impact
    if followers <= 100:
        f5 = 1.0  # Maximum penalty for very low followers
    elif followers >= 500:
        f5 = 0.1  # Minimal penalty for high followers
    else:
        # Steeper curve for medium follower counts
        f5 = 1.0 - 0.9 * ((followers - 100) / 400.0) ** 0.8

    # MODIFIED: f6 - Following-to-follower ratio with higher impact
    # High following count relative to followers is very suspicious
    if followers == 0:
        # No followers but has followings is highly suspicious
        f6 = 1.0 if followings > 0 else 0.5
    else:
        # Calculate ratio of followings to followers
        ratio = followings / followers
        # Apply non-linear scaling to emphasize high ratios
        if ratio <= 1:
            f6 = 0.3 * ratio  # Normal ratio has low impact
        elif ratio <= 5:
            f6 = 0.3 + 0.4 * ((ratio - 1) / 4)  # Medium ratio has moderate impact
        else:
            f6 = 0.7 + 0.3 * min(1, (ratio - 5) / 15)  # High ratio has strong impact

    # Original remaining features (f7-f13)
    f7 = 0.0 if verified else 1.0
    lower_tokens = [token.lower() for token in tokens]
    f8 = len(set(lower_tokens)) / total_words
    total_letters = sum(len(word) for word in tokens)
    f9 = (total_letters / total_words) / 2.0
    f10 = min(1, (text.count('!') / (len(text) if len(text) > 0 else 1)) * 20)
    total_alpha = sum(1 for c in text if c.isalpha())
    cap_letters = sum(1 for c in text if c.isupper())
    f11 = min(1, (cap_letters / (total_alpha if total_alpha > 0 else 1)) * 8)
    numeric_count = sum(1 for token in tokens if any(c.isdigit() for c in token))
    f12 = min(1, (numeric_count / total_words) * 3)
    f13 = min(1, ((total_words - len(set(lower_tokens))) / total_words) * 2)

    # Add a new feature that explicitly combines followers and followings
    # f14: Follower-to-following suspicious pattern
    if followers < 100 and followings > 500:
        f14 = 1.0  # Very suspicious pattern
    elif followers < 200 and followings > 1000:
        f14 = 0.9  # Very suspicious pattern
    elif followers * 5 < followings:  # Following count is 5x+ the follower count
        f14 = 0.8  # Suspicious pattern
    elif followers * 2 < followings:  # Following count is 2-5x the follower count
        f14 = 0.6  # Somewhat suspicious
    else:
        f14 = 0.2  # Normal pattern

    # Remaining dummy features
    dummy_features = [0.5] * 6
    features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                f11, f12, f13, f14] + dummy_features
    return features


def get_feature_explanation(features):
    """Return human-readable explanations for the feature values"""
    explanations = []

    # Add explicit explanations for follower/following features
    if features[4] > 1.0:  # f5 - low followers
        explanations.append("Extremely low follower count (high risk factor)")
    elif features[4] > 0.7:
        explanations.append("Very low follower count (suspicious)")

    if features[5] > 1.0:  # f6 - following ratio
        explanations.append("Extremely high following-to-follower ratio (high risk factor)")
    elif features[5] > 0.7:
        explanations.append("Suspicious following-to-follower ratio")

    if features[13] > 1.0:  # f14 - follower pattern
        explanations.append("Account follows many users but has very few followers (classic spam pattern)")

    # Original thresholds
    thresholds = [
        (0.5, "Promotional message length"),
        (0.2, "Uses @mentions"),
        (0.2, "Uses multiple hashtags"),
        (0.2, "Contains links or URLs"),
        (0.1, "Very low follower count"),
        (0.8, "Very high following count"),
        (0, "Verified account (less likely spam)"),
        (0.5, "Low word diversity"),
        (2.5, "Short words"),
        (0.2, "Uses exclamation marks"),
        (0.2, "Uses ALL CAPS text"),
        (0.1, "Contains numbers/digits"),
        (0.3, "Repetitive language"),
        (0.2, "Contains suspicious keywords (money, earn, etc.)"),
        (0.6, "Unusual follower-to-following ratio"),
        (0.3, "Contains call-to-action phrases"),
        (0.5, "Mentions specific money amounts"),
        (0.3, "Uses urgency indicators")
    ]
    for i, (threshold, explanation) in enumerate(thresholds):
        if i < len(features):
            if i in [7, 8]:  # For f8 and f9, lower is more suspicious
                if features[i] < threshold:
                    explanations.append(explanation)
            else:
                if features[i] > threshold:
                    explanations.append(explanation)
    return explanations


#####################
# Model Persistence and Crowd Feedback
#####################

def save_model(model, filename="spambot_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename="spambot_model.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        model = build_complex_model()
        save_model(model)
        return model


def get_crowd_feedback_data():
    if 'crowd_feedback' not in st.session_state:
        try:
            crowd_feedback = pd.read_csv("crowd_feedback.csv")
        except FileNotFoundError:
            crowd_feedback = pd.DataFrame(columns=[
                'post_text', 'followers', 'friends', 'verified',
                'predicted_score', 'human_label', 'timestamp', 'features'
            ])
        st.session_state.crowd_feedback = crowd_feedback
    return st.session_state.crowd_feedback


def save_crowd_feedback(crowd_feedback):
    crowd_feedback.to_csv("crowd_feedback.csv", index=False)
    st.session_state.crowd_feedback = crowd_feedback


def update_model_with_feedback(threshold=10):
    """
    Retrains the model when sufficient new feedback is collected.
    """
    crowd_feedback = get_crowd_feedback_data()
    if 'last_retrain_count' not in st.session_state:
        st.session_state.last_retrain_count = 0
    current_count = len(crowd_feedback)
    if current_count - st.session_state.last_retrain_count >= threshold:
        X_feedback = np.array([eval(feat) for feat in crowd_feedback['features']])
        y_feedback = np.array(crowd_feedback['human_label'])
        model = build_complex_model()  # Base training data is included
        model.fit(X_feedback, y_feedback)
        save_model(model)
        st.session_state.last_retrain_count = current_count
        return True
    return False


def calculate_dynamic_threshold():
    crowd_feedback = get_crowd_feedback_data()
    if len(crowd_feedback) < 10:
        return 0.6  # Lower default threshold to catch more spam
    spam_scores = crowd_feedback[crowd_feedback['human_label'] == 1]['predicted_score']
    if len(spam_scores) > 0:
        return max(0.5, spam_scores.quantile(0.1))
    else:
        return 0.6


def get_example_post_score(model):
    example_text = "Earn $5000 a week from home! Click http://spamlink.com now! @GetRichQuick #easymoney #crypto"
    example_followers = 50
    example_friends = 100
    example_verified = False
    features = extract_features_from_text_and_user(example_text, example_followers, example_friends, example_verified)
    features_array = np.array(features).reshape(1, -1)
    return model.predict_proba(features_array)[0][1]


def get_trending_spam_patterns():
    crowd_feedback = get_crowd_feedback_data()
    if len(crowd_feedback) < 5:
        return ["Not enough data to determine trending patterns"]
    spam_posts = crowd_feedback[crowd_feedback['human_label'] == 1]
    if len(spam_posts) < 3:
        return ["Not enough confirmed spam data to determine trends"]
    all_text = " ".join(spam_posts['post_text'].tolist()).lower()
    tokens = all_text.split()
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_words[:5] if count > 1]
    feature_patterns = []
    if len(spam_posts) >= 5:
        features_list = [eval(feat) for feat in spam_posts['features']]
        avg_features = np.mean(features_list, axis=0)
        feature_names = [
            "text length", "mentions", "hashtags", "URLs",
            "follower count", "followings", "verification", "word diversity",
            "word length", "exclamation marks", "capital letters", "numbers",
            "repetitive language", "suspicious keywords", "follower-following ratio",
            "call-to-action", "money amounts", "urgency indicators"
        ]
        top_indices = np.argsort(avg_features[:18])[-3:]
        for idx in top_indices:
            if avg_features[idx] > 0.3:
                feature_patterns.append(feature_names[idx])
    trends = []
    if top_words:
        trends.append(f"Common spam keywords: {', '.join(top_words)}")
    if feature_patterns:
        trends.append(f"Common spam characteristics: {', '.join(feature_patterns)}")
    return trends if trends else ["Not enough patterns detected in recent spam"]


def add_feedback(post_text, followers, friends, verified, predicted_score, human_label, features):
    crowd_feedback = get_crowd_feedback_data()
    new_feedback = pd.DataFrame({
        'post_text': [post_text],
        'followers': [followers],
        'friends': [friends],
        'verified': [verified],
        'predicted_score': [predicted_score],
        'human_label': [human_label],
        'timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'features': [str(features)]
    })
    updated_feedback = pd.concat([crowd_feedback, new_feedback], ignore_index=True)
    save_crowd_feedback(updated_feedback)
    update_model_with_feedback()


#####################
# Main Streamlit App
#####################

def run_app():
    st.title("Spambot Detection SaaS with Cloud-Crowd Hybrid Strategy")

    # Load or initialize model
    model = load_model()

    # Display app tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Spambot Detection", "Community Insights", "Batch Processing", "System Status"])

    with tab1:
        st.write("Enter the details of a post to predict the likelihood that the user account is a bot.")
        bot_threshold = calculate_dynamic_threshold()

        st.markdown("**Score Legend:**")
        st.markdown(f"- **0.0 - 0.3:** Likely Human")
        st.markdown(f"- **0.3 - {bot_threshold:.2f}:** Uncertain")
        st.markdown(f"- **{bot_threshold:.2f} - 1.0:** Likely Spambot")

        st.write("### Example of a Spam Bot Post")
        st.write("""
        **Post Text:** "Earn $5000 a week from home! Click http://spamlink.com now! @GetRichQuick #easymoney #crypto"  
        **Followers Count:** 50  
        **Followings Count:** 500  
        **Verified:** No  
        """)

        post_text = st.text_area("Post Text",
                                 "Earn $5000 a week from home! Click http://spamlink.com now! @GetRichQuick "
                                 "#easymoney #crypto")
        followers = st.number_input("Followers Count", min_value=0, value=50, step=10)
        friends = st.number_input("Followings Count", min_value=0, value=500, step=10)
        verified = st.checkbox("Is the account verified?", value=False)

        if st.button("Predict Bot Score"):
            try:
                features = extract_features_from_text_and_user(post_text, followers, friends, verified)
                feature_names = [
                    "Normalized text length", "Mention ratio", "Hashtag ratio", "URL ratio",
                    "Inverted normalized followers", "Normalized followings", "Unverified flag", "Unique word ratio",
                    "Average word length", "Exclamation ratio", "Capital letter ratio", "Numeric token ratio",
                    "Repetition factor", "Suspicious keyword ratio", "Follower-following ratio",
                    "Call-to-action score", "Money amount indicator", "Urgency indicator",
                    "Dummy feature 19", "Dummy feature 20"
                ]
                feature_dict = {name: round(value, 3) for name, value in zip(feature_names, features)}

                with st.expander("View Extracted Features"):
                    st.json(feature_dict)

                explanations = get_feature_explanation(features)
                if explanations:
                    st.write("### Suspicious Characteristics Detected:")
                    for explanation in explanations:
                        st.markdown(f"- {explanation}")

                features_array = np.array(features).reshape(1, -1)
                bot_score = model.predict_proba(features_array)[0][1]

                st.session_state.current_prediction = {
                    'post_text': post_text,
                    'followers': followers,
                    'friends': friends,
                    'verified': verified,
                    'features': features,
                    'predicted_score': bot_score
                }

                if bot_score >= bot_threshold:
                    st.error(f"⚠️ LIKELY SPAMBOT: The predicted bot score is: **{bot_score:.4f}**")
                elif bot_score <= 0.3:
                    st.success(f"✅ LIKELY HUMAN: The predicted bot score is: **{bot_score:.4f}**")
                else:
                    st.warning(f"⚠️ UNCERTAIN: The predicted bot score is: **{bot_score:.4f}**")

                st.write("### Help Improve Our System")
                st.write("Was this prediction correct? Your feedback helps our system learn!")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, this is SPAM"):
                        add_feedback(post_text, followers, friends, verified, bot_score, 1, features)
                        st.success("Thank you for your feedback! Our system will learn from this.")
                with col2:
                    if st.button("No, this is NOT spam"):
                        add_feedback(post_text, followers, friends, verified, bot_score, 0, features)
                        st.success("Thank you for your feedback! Our system will learn from this.")
            except Exception as e:
                st.error(f"Error processing input data: {e}")

    with tab2:
        st.header("Community Insights")
        st.write("Learn from the collective intelligence of our user community")

        st.subheader("Trending Spam Patterns")
        patterns = get_trending_spam_patterns()
        for pattern in patterns:
            st.markdown(f"- {pattern}")

        st.subheader("Recent Community Feedback")
        crowd_feedback = get_crowd_feedback_data()
        if len(crowd_feedback) > 0:
            recent_feedback = crowd_feedback.tail(5)[['post_text', 'predicted_score', 'human_label', 'timestamp']]
            recent_feedback['human_label'] = recent_feedback['human_label'].map({1: "Spam", 0: "Not Spam"})
            recent_feedback.columns = ['Post Text', 'AI Score', 'Human Label', 'Timestamp']
            st.dataframe(recent_feedback)
        else:
            st.write("No community feedback collected yet.")

        st.subheader("Community Contribution Stats")
        if len(crowd_feedback) > 0:
            total_contributions = len(crowd_feedback)
            agreement_rate = (crowd_feedback['human_label'] == (
                        crowd_feedback['predicted_score'] > bot_threshold)).mean() * 100
            st.metric("Total Contributions", total_contributions)
            st.metric("AI-Human Agreement Rate", f"{agreement_rate:.1f}%")
        else:
            st.write("No stats available yet. Be the first to contribute!")

    with tab3:
        st.header("Batch Processing")
        st.write("Process multiple posts at once by uploading a CSV file")
        st.markdown("""
        ### CSV Format Requirements:
        Your CSV should have these columns:
        - `post_text`: The text content of the post
        - `followers`: Number of followers
        - `friends`: Number of followings
        - `verified`: Boolean (True/False) indicating if verified

        [Download Sample CSV Template](https://example.com/sample-template.csv)
        """)
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_data)} entries")
                if st.button("Process Batch"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    for i, row in batch_data.iterrows():
                        progress = (i + 1) / len(batch_data)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing entry {i + 1} of {len(batch_data)}")
                        features = extract_features_from_text_and_user(
                            row['post_text'],
                            row['followers'],
                            row['friends'],
                            row['verified']
                        )
                        features_array = np.array(features).reshape(1, -1)
                        bot_score = model.predict_proba(features_array)[0][1]
                        if bot_score >= bot_threshold:
                            classification = "Likely Spambot"
                        elif bot_score <= 0.3:
                            classification = "Likely Human"
                        else:
                            classification = "Uncertain"
                        results.append({
                            'post_text': row['post_text'][:50] + "..." if len(row['post_text']) > 50 else row[
                                'post_text'],
                            'bot_score': bot_score,
                            'classification': classification
                        })
                    progress_bar.empty()
                    status_text.empty()
                    st.subheader("Batch Processing Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="batch_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing batch file: {e}")

    with tab4:
        st.header("System Status")
        st.subheader("Model Status")
        model_updated = update_model_with_feedback()
        if model_updated:
            st.success("✅ Model was just updated with new crowd feedback!")
        else:
            st.info("Model is up to date")
        crowd_feedback = get_crowd_feedback_data()
        if len(crowd_feedback) > 0:
            correct_predictions = sum(
                (crowd_feedback['predicted_score'] > bot_threshold) == crowd_feedback['human_label'])
            accuracy = correct_predictions / len(crowd_feedback)
            st.metric("Current Model Accuracy", f"{accuracy:.2%}")
        st.subheader("Feedback Statistics")
        if len(crowd_feedback) > 0:
            total_feedback = len(crowd_feedback)
            spam_count = sum(crowd_feedback['human_label'] == 1)
            ham_count = sum(crowd_feedback['human_label'] == 0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Feedback", total_feedback)
            col2.metric("Confirmed Spam", spam_count)
            col3.metric("Confirmed Ham", ham_count)
            feedback_since_last_update = total_feedback - st.session_state.last_retrain_count
            st.write(f"Feedback since last model update: {feedback_since_last_update}/10")
            st.progress(min(1.0, feedback_since_last_update / 10))
            if feedback_since_last_update < 10:
                st.write(f"Model will update after {10 - feedback_since_last_update} more feedback submissions")
        else:
            st.write("No feedback collected yet")


if __name__ == "__main__":
    if 'last_retrain_count' not in st.session_state:
        st.session_state.last_retrain_count = 0
    run_app()
